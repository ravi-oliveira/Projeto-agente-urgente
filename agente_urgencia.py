import ollama
import chromadb
import json
import os
import datetime
import threading
import customtkinter as ctk

import sys

BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
PASTA_BANCO_RAG = os.path.join(BASE_DIR, "banco_rag_saude")
ARQUIVO_BANCO_FILA = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "fila_triagem.json")

# --- CONFIGURAÇÕES DO SISTEMA ---
MODELO_LLM = "qwen2.5:14b"
MODELO_EMBED = "bge-m3"
PASTA_BANCO_RAG = "banco_rag_saude"
ARQUIVO_BANCO_FILA = "fila_triagem.json"

# Tabela Oficial de Classificação de Risco (Protocolo de Manchester / UPA)
PROTOCOLO_MANCHESTER = [
    {
        "nivel": 1,
        "sigla": "VM",
        "cor_nome": "Vermelho",
        "classificacao": "Emergência",
        "tempo": "Imediato (0 min)",
        "exemplos": "Risco iminente de morte, parada cardiorrespiratória, falta de ar grave, dor precordial ou choque.",
        "hex": "#DC2626",
        "hex_suave": "#FEF2F2"
    },
    {
        "nivel": 2,
        "sigla": "LR",
        "cor_nome": "Laranja",
        "classificacao": "Muito urgente",
        "tempo": "Até 10 min (15–20 min)",
        "exemplos": "Risco grave, dor intensa (8-10/10), sangramento importante, alteração de consciência ou sinais vitais.",
        "hex": "#EA580C",
        "hex_suave": "#FFF7ED"
    },
    {
        "nivel": 3,
        "sigla": "AM",
        "cor_nome": "Amarelo",
        "classificacao": "Urgente",
        "tempo": "Até 60 minutos",
        "exemplos": "Risco moderado, dor forte controlável (5-7/10), vômitos persistentes, febre alta ou suspeita de fratura.",
        "hex": "#CA8A04",
        "hex_suave": "#FEFCE8"
    },
    {
        "nivel": 4,
        "sigla": "VD",
        "cor_nome": "Verde",
        "classificacao": "Pouco urgente",
        "tempo": "Até 120 minutos",
        "exemplos": "Baixa gravidade, sintomas leves/estáveis (dor 1-4/10), dor de cabeça leve ou infecção sem sinais de alarme.",
        "hex": "#16A34A",
        "hex_suave": "#F0FDF4"
    },
    {
        "nivel": 5,
        "sigla": "AZ",
        "cor_nome": "Azul",
        "classificacao": "Não urgente",
        "tempo": "Até 240 min (ou UBS)",
        "exemplos": "Quadros crônicos sem agudização, queixas antigas ou demandas eletivas, como troca de receitas.",
        "hex": "#2563EB",
        "hex_suave": "#EFF6FF"
    }
]

MAPA_CORES = {item["cor_nome"].upper(): item for item in PROTOCOLO_MANCHESTER}

PESOS_PRIORIDADE = {
    "GESTANTE": (1, "G", "Gestante"),
    "IDOSO": (2, "I", "Idoso (60+)"),
    "CRIANCA": (3, "C", "Criança / Lactente"),
    "PCD": (4, "P", "Pessoa com Deficiência"),
    "NORMAL": (5, "N", "Atendimento Normal")
}


# --- MOTOR RAG CLÍNICO LOCAL (CHROMADB + BGE-M3) ---

class MotorRAGClinico:
    def __init__(self, caminho_banco=PASTA_BANCO_RAG):
        self.ativo = False
        self.colecao = None
        if os.path.exists(caminho_banco):
            try:
                cliente = chromadb.PersistentClient(path=caminho_banco)
                self.colecao = cliente.get_collection("protocolos_urgencia")
                if self.colecao.count() > 0:
                    self.ativo = True
            except Exception:
                self.ativo = False

    def buscar_evidencias(self, relato_paciente):
        """
        Consulta os 3 documentos vetorizados separadamente para garantir que o LLM receba:
        - Descritores de cor (Protocolo UPA)
        - Sinais de alerta e vulnerabilidades (Caderno 28 MS)
        - Código da queixa (CIAP-2 Brasil)
        """
        if not self.ativo or not self.colecao:
            return ""

        try:
            resp = ollama.embed(model=MODELO_EMBED, input=[relato_paciente])
            vetor_query = resp["embeddings"][0]

            blocos = []
            fontes_alvo = [
                ("Protocolo_UPA", "PROTOCOLO UPA (DESCRITORES DE COR)", 2),
                ("Caderno28_MS", "CADERNO 28 MS (SINAIS DE ALERTA)", 2),
                ("CIAP2_Brasil", "CIAP-2 BRASIL (CÓDIGO DA QUEIXA)", 2)
            ]

            for chave_fonte, rotulo, k in fontes_alvo:
                res = self.colecao.query(
                    query_embeddings=[vetor_query],
                    n_results=k,
                    where={"fonte": chave_fonte}
                )
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                for d, m in zip(docs, metas):
                    blocos.append(f"[{rotulo} | pág. {m.get('pagina', '?')}]: {d[:700]}")

            return "\n\n".join(blocos)
        except Exception:
            return ""


# --- CAMADA DE DADOS E ORDENAÇÃO DETERMINÍSTICA ---

class GerenciadorFila:
    def __init__(self, caminho_arquivo=ARQUIVO_BANCO_FILA):
        self.caminho = caminho_arquivo
        self.pacientes = self._carregar()

    def _carregar(self):
        if os.path.exists(self.caminho):
            try:
                with open(self.caminho, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _salvar(self):
        with open(self.caminho, "w", encoding="utf-8") as f:
            json.dump(self.pacientes, f, ensure_ascii=False, indent=2)

    def registrar_paciente(self, dados_triagem):
        cor_key = dados_triagem.get("cor", "VERDE").strip().upper()
        if cor_key not in MAPA_CORES:
            cor_key = "VERDE"
        info_cor = MAPA_CORES[cor_key]

        prio_key = dados_triagem.get("prioridade_legal", "NORMAL").strip().upper()
        # Trava determinística de Gênero e Idade
        genero = dados_triagem.get("genero", "Não informado")
        if prio_key == "GESTANTE":
            genero = "Feminino"

        idade_str = str(dados_triagem.get("idade", "Não informada"))
        digitos_idade = "".join(c for c in idade_str if c.isdigit())
        if digitos_idade and prio_key == "NORMAL":
            idade_num = int(digitos_idade)
            if idade_num >= 60:
                prio_key = "IDOSO"
            elif idade_num < 12:
                prio_key = "CRIANCA"
        if prio_key not in PESOS_PRIORIDADE:
            prio_key = "NORMAL"
        peso_prio, letra_prio, rotulo_prio = PESOS_PRIORIDADE[prio_key]

        numero_seq = len(self.pacientes) + 1
        senha = f"{info_cor['sigla']}-{letra_prio}{numero_seq:03d}"
        agora = datetime.datetime.now()

        registro = {
            "senha": senha,
            "timestamp": agora.timestamp(),
            "hora_entrada": agora.strftime("%H:%M:%S"),
            "data": agora.strftime("%d/%m/%Y"),
            "cor": info_cor["cor_nome"],
            "nivel_clinico": info_cor["nivel"],
            "classificacao": info_cor["classificacao"],
            "tempo_meta": info_cor["tempo"],
            "prioridade_legal": rotulo_prio,
            "peso_prioridade": peso_prio,
            "idade": dados_triagem.get("idade", "Não informada"),
            "genero": genero,
            "peso_kg": dados_triagem.get("peso_kg", "Não informado"),
            "altura": dados_triagem.get("altura", "Não informada"),
            "escala_dor_0_10": dados_triagem.get("escala_dor_0_10", "N/A"),
            "sintomas_correlatos": dados_triagem.get("sintomas_correlatos", "Nenhum"),
            "habitos_risco": dados_triagem.get("habitos_risco", "Não informado"),
            "codigo_ciap2": dados_triagem.get("codigo_ciap2", "Não classificado"),
            "queixa_resumida": dados_triagem.get("queixa_resumida", ""),
            "justificativa_clinica": dados_triagem.get("justificativa_clinica", "")
        }

        self.pacientes.append(registro)
        self._salvar()
        return registro

    def obter_fila_ordenada(self):
        return sorted(
            self.pacientes,
            key=lambda p: (p["nivel_clinico"], p["peso_prioridade"], p["timestamp"])
        )

    def limpar_fila(self):
        self.pacientes = []
        self._salvar()


# --- PROMPT CLÍNICO ORIENTADO POR RAG ---

SYSTEM_PROMPT_TRIAGEM = """
Você é um atendente de acolhimento humano, educado e objetivo de um Pronto-Socorro.
Conduza a conversa de forma natural, fazendo no máximo 1 ou 2 perguntas curtas por vez.

REGRAS CRÍTICAS DE COMPORTAMENTO:
- NUNCA mencione termos técnicos ao paciente (NUNCA fale "CIAP-2", "Protocolo de Manchester", "prioridade legal" ou "codificar").
- NUNCA pergunte se o paciente é criança ou idoso — você DEVE deduzir isso matematicamente pela idade informada (menor de 12 anos = CRIANCA; 60 anos ou mais = IDOSO).
- NUNCA repita uma pergunta que o paciente já respondeu.

FLUXO OBRIGATÓRIO DE PERGUNTAS (SIGA ESTA ORDEM, EXCETO SE FOR EMERGÊNCIA VERMELHO):
1. IDENTIFICAÇÃO E CONDIÇÃO ESPECIAL: Antes de tudo, saiba para quem é o atendimento (para o próprio usuário ou acompanhante) e se o paciente é GESTANTE ou Pessoa com Deficiência (PcD).
2. IDADE E GÊNERO:
   - Pergunte a idade do paciente.
   - Logo após saber a idade, pergunte o gênero do paciente.
   - EXCEÇÃO OBRIGATÓRIA: Se o paciente informou que é GESTANTE, PULE a pergunta de gênero imediatamente e registre "genero": "Feminino".
3. QUEIXA, ESCALA E SINTOMAS CORRELATOS:
   - Entenda a queixa principal e há quanto tempo começou.
   - Peça uma nota de 0 a 10 para a intensidade da dor ou sintoma.
   - Pergunte se há sintomas correlatos (febre, náusea, tontura, visão turva, falta de ar, etc.).
4. BIOMETRIA E HÁBITOS:
   - Pergunte peso aproximado, altura aproximada e se há hábitos de risco (sedentarismo, tabagismo, álcool/substâncias) ou doenças crônicas.
5. CONFIRMAÇÃO FINAL (OBRIGATÓRIA ANTES DE CONCLUIR):
   - Quando tiver coletado todas as etapas acima, NÃO faça resumo técnico na conversa.
   - Mantenha "status": "COLETANDO" e pergunte EXATAMENTE:
     "Mais alguma informação relevante que deve ser informada? Se não, digite 'sim' para continuar."
6. ENCERRAMENTO:
   - Assim que o paciente responder à confirmação final (ex: "sim", "não", "só isso", "pode continuar"), mude "status" para "CONCLUIDO", preencha todos os campos de "dados_triagem" e envie apenas uma mensagem curta:
     "Obrigado! Sua ficha de espera e classificação foram geradas abaixo."

CLASSIFICAÇÃO DE RISCO (USO INTERNO PARA O JSON):
- VERMELHO (0 min): Risco iminente de morte, falta de ar grave, dor no peito em aperto/irradiando, inconsciência, convulsão ativa, rigidez na nuca com febre ou "pior dor de cabeça da vida". (Em caso de VERMELHO, pule o restante do fluxo e conclua imediatamente).
- LARANJA (10 min): Dor intensa (8 a 10/10), sangramento importante, febre > 40°C, trauma craniano recente.
- AMARELO (60 min): Dor moderada (5 a 7/10), febre > 38,5°C, vômitos persistentes, crise hipertensiva com sintomas, suspeita de fratura.
- VERDE (120 min): Dor leve (1 a 4/10), sintomas leves/estáveis sem sinais de alarme.
- AZUL (240 min): Quadros crônicos estáveis, troca de receitas, atestados.

RESPONDA SEMPRE E APENAS EM JSON VÁLIDO NESTE FORMATO:
{
  "pensamento_interno": "Etapa atual do fluxo (1 a 6) | Dados já coletados | O que falta perguntar | Checagem: se gestante, pular gênero; se já tem idade, deduzir criança/idoso sem perguntar.",
  "status": "COLETANDO",
  "mensagem_paciente": "Mensagem curta, humana e sem jargões técnicos.",
  "dados_triagem": {
    "cor": "VERDE",
    "prioridade_legal": "NORMAL",
    "idade": "Não informada",
    "genero": "Não informado",
    "peso_kg": "Não informado",
    "altura": "Não informada",
    "escala_dor_0_10": "Não aplicável",
    "sintomas_correlatos": "Nenhum relatado",
    "habitos_risco": "Não informado",
    "codigo_ciap2": "Ex: N01 - Cefaleia",
    "queixa_resumida": "",
    "justificativa_clinica": ""
  }
}
"""


# --- INTERFACE GRÁFICA CLÍNICA LIMPA (5 CORES VISÍVEIS) ---

class SistemaTriagemApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("light")
        self.title("Acolhimento e Classificação de Risco — Pronto-Socorro")
        self.geometry("1260x780")
        self.minsize(1060, 680)
        self.configure(fg_color="#F8FAFC")

        self.gerenciador = GerenciadorFila()
        self.rag = MotorRAGClinico()
        self.historico_llm = []
        self.relatos_paciente = []
        self.processando = False
        self.atendimento_concluido = False

        self.grid_columnconfigure(0, weight=5)
        self.grid_columnconfigure(1, weight=6)
        self.grid_rowconfigure(1, weight=1)

        self._construir_cabecalho()
        self._construir_painel_tabela()
        self._construir_painel_atendimento()
        self._iniciar_novo_atendimento()

    def _construir_cabecalho(self):
        header = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=0, height=62, border_width=1, border_color="#E2E8F0")
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        lbl_titulo = ctk.CTkLabel(
            header,
            text="  ✚  PRONTO-SOCORRO  |  TRIAGEM E CLASSIFICAÇÃO DE RISCO",
            font=ctk.CTkFont(family="Ubuntu", size=17, weight="bold"),
            text_color="#0F172A",
            anchor="w"
        )
        lbl_titulo.grid(row=0, column=0, padx=20, pady=15, sticky="w")

        ctk.CTkButton(
            header,
            text="Fila Ordenada (Pesquisa)",
            width=180,
            height=32,
            corner_radius=8,
            fg_color="#F1F5F9",
            hover_color="#E2E8F0",
            text_color="#334155",
            border_width=1,
            border_color="#CBD5E1",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._abrir_janela_fila
        ).grid(row=0, column=1, padx=20, pady=12, sticky="e")

    def _construir_painel_tabela(self):
        """Usa grid com peso igual para garantir que TODAS as 5 cores caibam na tela."""
        container = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=14, border_width=1, border_color="#E2E8F0")
        container.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=16)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(2, weight=1)  # Faz a área das 5 cores ocupar todo o espaço livre

        ctk.CTkLabel(
            container,
            text="CLASSIFICAÇÃO DE RISCO (PROTOCOLO DE MANCHESTER)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#0F172A",
            anchor="w"
        ).grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 2))

        ctk.CTkLabel(
            container,
            text="Atendimento organizado por gravidade clínica. Prioridades legais têm preferência dentro da mesma cor.",
            font=ctk.CTkFont(size=11),
            text_color="#64748B",
            wraplength=460,
            justify="left",
            anchor="w"
        ).grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 8))

        # Grade fixa de 5 linhas iguais (uma para cada cor)
        grade_cores = ctk.CTkFrame(container, fg_color="transparent")
        grade_cores.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 12))
        grade_cores.grid_columnconfigure(0, weight=1)

        for i, item in enumerate(PROTOCOLO_MANCHESTER):
            # weight=1 em cada linha obriga o Python a dividir a altura em 5 partes iguais
            grade_cores.grid_rowconfigure(i, weight=1)

            card = ctk.CTkFrame(
                grade_cores,
                fg_color=item["hex_suave"],
                corner_radius=10,
                border_width=1,
                border_color="#E2E8F0"
            )
            card.grid(row=i, column=0, sticky="nsew", pady=4)
            card.grid_columnconfigure(1, weight=1)
            card.grid_rowconfigure(1, weight=1)

            # Faixa lateral colorida
            faixa = ctk.CTkFrame(card, fg_color=item["hex"], width=8, corner_radius=4)
            faixa.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(6, 10), pady=6)

            # Linha 1 do card: Cor + Tempo meta
            topo = ctk.CTkFrame(card, fg_color="transparent")
            topo.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(6, 0))

            ctk.CTkLabel(
                topo,
                text=f"{item['cor_nome'].upper()} — {item['classificacao']}",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=item["hex"]
            ).pack(side="left")

            ctk.CTkLabel(
                topo,
                text=item["tempo"],
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#334155"
            ).pack(side="right")

            # Linha 2 do card: Exemplos clínicos
            ctk.CTkLabel(
                card,
                text=item["exemplos"],
                font=ctk.CTkFont(size=11),
                text_color="#475569",
                wraplength=430,
                justify="left",
                anchor="nw"
            ).grid(row=1, column=1, sticky="nsew", padx=(0, 10), pady=(2, 6))

    def _construir_painel_atendimento(self):
        container = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=14, border_width=1, border_color="#E2E8F0")
        container.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=18)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)

        topo_chat = ctk.CTkFrame(container, fg_color="transparent")
        topo_chat.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 8))

        ctk.CTkLabel(
            topo_chat,
            text="ATENDIMENTO DE TRIAGEM",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#0F172A"
        ).pack(side="left")

        ctk.CTkButton(
            topo_chat,
            text="↻ Novo Paciente",
            width=125,
            height=28,
            corner_radius=6,
            fg_color="#F1F5F9",
            hover_color="#E2E8F0",
            text_color="#0F172A",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._iniciar_novo_atendimento
        ).pack(side="right")

        self.chat_scroll = ctk.CTkScrollableFrame(
            container,
            fg_color="#F8FAFC",
            corner_radius=10,
            border_width=1,
            border_color="#E2E8F0"
        )
        self.chat_scroll.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 10))
        self.chat_scroll.grid_columnconfigure(0, weight=1)

        rodape = ctk.CTkFrame(container, fg_color="transparent")
        rodape.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 16))
        rodape.grid_columnconfigure(0, weight=1)

        self.lbl_status = ctk.CTkLabel(
            rodape,
            text="Pronto para acolhimento.",
            font=ctk.CTkFont(size=11),
            text_color="#64748B",
            anchor="w"
        )
        self.lbl_status.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        self.entrada_msg = ctk.CTkEntry(
            rodape,
            placeholder_text="Descreva o que está sentindo, sua idade e se é gestante/idoso/criança...",
            height=42,
            fg_color="#FFFFFF",
            border_color="#CBD5E1",
            text_color="#0F172A",
            font=ctk.CTkFont(size=13),
            corner_radius=8
        )
        self.entrada_msg.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.entrada_msg.bind("<Return>", lambda e: self._enviar_mensagem())

        self.btn_enviar = ctk.CTkButton(
            rodape,
            text="Enviar",
            width=100,
            height=42,
            corner_radius=8,
            fg_color="#0F172A",
            hover_color="#334155",
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._enviar_mensagem
        )
        self.btn_enviar.grid(row=1, column=1)

    # --- FLUXO DE CONVERSA E TRIAGEM ---

    def _limpar_chat_ui(self):
        for widget in self.chat_scroll.winfo_children():
            widget.destroy()

    def _rolar_fim(self):
        self.after(80, lambda: self.chat_scroll._parent_canvas.yview_moveto(1.0))

    def _iniciar_novo_atendimento(self):
        if self.processando:
            return
        self._limpar_chat_ui()
        self.atendimento_concluido = False
        self.relatos_paciente = []
        self.entrada_msg.configure(state="normal")
        self.btn_enviar.configure(state="normal")
        self.lbl_status.configure(text="Pronto para acolhimento.")

        self.historico_llm = [{"role": "system", "content": SYSTEM_PROMPT_TRIAGEM}]
        saudacao = (
            "Olá! Bem-vindo ao acolhimento do Pronto-Socorro.\n"
            "Antes de começarmos, o atendimento é para você ou para outra pessoa? "
            "O paciente é gestante ou pessoa com deficiência (PcD)?"
        )
        self.historico_llm.append({
            "role": "assistant",
            "content": json.dumps({"status": "COLETANDO", "mensagem_paciente": saudacao, "dados_triagem": {}})
        })
        self._adicionar_bolha("ATENDENTE DE TRIAGEM", saudacao, eh_usuario=False)

    def _adicionar_bolha(self, autor, texto, eh_usuario=False):
        wrapper = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        wrapper.pack(fill="x", pady=5, padx=6)

        cor_bg = "#0F172A" if eh_usuario else "#FFFFFF"
        cor_txt = "#FFFFFF" if eh_usuario else "#1E293B"
        cor_autor = "#94A3B8" if eh_usuario else "#2563EB"
        lado = "right" if eh_usuario else "left"

        bolha = ctk.CTkFrame(
            wrapper,
            fg_color=cor_bg,
            corner_radius=12,
            border_width=0 if eh_usuario else 1,
            border_color="#E2E8F0"
        )
        bolha.pack(side=lado, padx=4)

        ctk.CTkLabel(
            bolha,
            text=autor,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=cor_autor,
            anchor="e" if eh_usuario else "w"
        ).pack(fill="x", padx=14, pady=(8, 0))

        ctk.CTkLabel(
            bolha,
            text=texto,
            font=ctk.CTkFont(size=13),
            text_color=cor_txt,
            wraplength=460,
            justify="left"
        ).pack(padx=14, pady=(2, 10))
        self._rolar_fim()

    def _exibir_cartao_senha(self, registro):
        info_cor = MAPA_CORES[registro["cor"].upper()]

        card = ctk.CTkFrame(
            self.chat_scroll,
            fg_color=info_cor["hex_suave"],
            corner_radius=12,
            border_width=2,
            border_color=info_cor["hex"]
        )
        card.pack(fill="x", pady=12, padx=10)

        topo = ctk.CTkFrame(card, fg_color=info_cor["hex"], corner_radius=8)
        topo.pack(fill="x", padx=12, pady=(12, 8))

        ctk.CTkLabel(
            topo,
            text=f"SENHA GERADA:  {registro['senha']}  —  {registro['cor'].upper()} ({registro['classificacao'].upper()})",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFFFFF"
        ).pack(pady=8)

        detalhes = (
            f"Tempo Meta: {registro['tempo_meta']}  |  CIAP-2: {registro['codigo_ciap2']}\n"
            f"Prioridade Legal: {registro['prioridade_legal']}  |  Idade: {registro['idade']}  |  Gênero: {registro.get('genero', 'Não informado')}\n"
            f"Biometria: {registro['peso_kg']} / {registro['altura']}  |  Escala (0-10): {registro['escala_dor_0_10']}\n"
            f"Queixa Principal: {registro['queixa_resumida']}\n"
            f"Sintomas Correlatos: {registro['sintomas_correlatos']}\n"
            f"Hábitos e Fatores de Risco: {registro['habitos_risco']}\n"
            f"Critério Clínico: {registro['justificativa_clinica']}\n"
            f"Registro: {registro['hora_entrada']} ({registro['data']})"
        )
        ctk.CTkLabel(
            card,
            text=detalhes,
            font=ctk.CTkFont(size=12),
            text_color="#1E293B",
            wraplength=510,
            justify="left",
            anchor="w"
        ).pack(fill="x", padx=16, pady=(2, 14))
        self._rolar_fim()

    def _enviar_mensagem(self):
        if self.processando or self.atendimento_concluido:
            return
        texto = self.entrada_msg.get().strip()
        if not texto:
            return

        self.entrada_msg.delete(0, "end")
        self._adicionar_bolha("PACIENTE", texto, eh_usuario=True)
        self.relatos_paciente.append(texto)

        self.processando = True
        self.btn_enviar.configure(state="disabled")
        self.lbl_status.configure(text="Consultando protocolos clínicos (RAG) e avaliando risco...")

        threading.Thread(target=self._processar_triagem_ia, args=(texto,), daemon=True).start()

    def _processar_triagem_ia(self, texto):
        try:
            # 1. Adiciona a fala atual do paciente na memória da conversa
            self.historico_llm.append({"role": "user", "content": texto})

            # 2. Consulta os PDFs vetorizados usando o MotorRAGClinico da sua classe
            relato_acumulado = " ".join(self.relatos_paciente)
            contexto_rag = self.rag.buscar_evidencias(relato_acumulado)

            mensagens_envio = list(self.historico_llm)
            if contexto_rag:
                mensagens_envio.insert(-1, {
                    "role": "system",
                    "content": (
                        "BASE CLÍNICA DE REFERÊNCIA (Protocolo Maringá, Caderno 28 e CIAP-2) "
                        "para apoiar sua decisão de risco e codificação CIAP-2:\n"
                        f"{contexto_rag}"
                    )
                })

            # 3. Envia para o Ollama
            resp = ollama.chat(
                model=MODELO_LLM,
                messages=mensagens_envio,
                format="json"
            )
            conteudo_bruto = resp["message"]["content"]
            self.historico_llm.append({"role": "assistant", "content": conteudo_bruto})

            # 4. Interpreta o JSON e exibe o Thinking no terminal
            dados = json.loads(conteudo_bruto)
            pensamento = dados.get("pensamento_interno", "")
            if pensamento:
                print(f"\n[THINKING IA] -> {pensamento}")

            status = dados.get("status", "COLETANDO").upper()
            msg_paciente = dados.get("mensagem_paciente", "Poderia me dar mais detalhes sobre o que está sentindo?")

            self.after(0, lambda: self._adicionar_bolha("ATENDENTE DE TRIAGEM", msg_paciente, eh_usuario=False))

            if status == "CONCLUIDO":
                dados_triagem = dados.get("dados_triagem", {})
                registro = self.gerenciador.registrar_paciente(dados_triagem)
                self.atendimento_concluido = True
                self.after(0, lambda r=registro: self._exibir_cartao_senha(r))
                self.after(0, lambda: self.entrada_msg.configure(state="disabled"))
                self.after(0, lambda: self.lbl_status.configure(
                    text="Triagem concluída e registrada na fila. Clique em 'Novo Paciente' para o próximo atendimento."
                ))
            else:
                self.after(0, lambda: self.lbl_status.configure(text="Aguardando resposta do paciente..."))

        except Exception as e:
            self.after(0, lambda err=str(e): self._adicionar_bolha(
                "SISTEMA", f"Erro ao consultar modelo local ({MODELO_LLM}): {err}", eh_usuario=False
            ))
        finally:
            self.processando = False
            if not self.atendimento_concluido:
                self.after(0, lambda: self.btn_enviar.configure(state="normal"))

    # --- JANELA DE MONITORAMENTO PARA PESQUISADORES ---

    def _abrir_janela_fila(self):
        janela = ctk.CTkToplevel(self)
        janela.title("Monitor de Fila Prioritária — Visão para Pesquisadores")
        janela.geometry("860x540")
        janela.configure(fg_color="#F8FAFC")
        janela.grab_set()

        topo = ctk.CTkFrame(janela, fg_color="#FFFFFF", corner_radius=0, border_width=1, border_color="#E2E8F0")
        topo.pack(fill="x")

        ctk.CTkLabel(
            topo,
            text="ORDEM DINÂMICA DE CHAMADA (GRAVIDADE CLÍNICA > PRIORIDADE LEGAL > CHEGADA)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#0F172A"
        ).pack(side="left", padx=16, pady=12)

        def limpar_e_fechar():
            self.gerenciador.limpar_fila()
            janela.destroy()

        ctk.CTkButton(
            topo,
            text="Limpar Fila",
            width=90,
            height=28,
            fg_color="#FEE2E2",
            hover_color="#FECACA",
            text_color="#B91C1C",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=limpar_e_fechar
        ).pack(side="right", padx=16, pady=10)

        lista_scroll = ctk.CTkScrollableFrame(janela, fg_color="#F8FAFC")
        lista_scroll.pack(fill="both", expand=True, padx=16, pady=14)

        fila = self.gerenciador.obter_fila_ordenada()
        if not fila:
            ctk.CTkLabel(
                lista_scroll,
                text="Nenhum paciente registrado na fila ainda.",
                font=ctk.CTkFont(size=13),
                text_color="#64748B"
            ).pack(pady=40)
            return

        for pos, p in enumerate(fila, 1):
            cor_hex = MAPA_CORES[p["cor"].upper()]["hex"]
            item_frame = ctk.CTkFrame(lista_scroll, fg_color="#FFFFFF", corner_radius=8, border_width=1, border_color="#E2E8F0")
            item_frame.pack(fill="x", pady=4)
            item_frame.grid_columnconfigure(1, weight=1)

            ctk.CTkFrame(item_frame, fg_color=cor_hex, width=8, height=10, corner_radius=4).grid(
                row=0, column=0, sticky="ns", padx=(6, 10), pady=6
            )

            texto_linha = (
                f"{pos}º  |  Senha: {p['senha']}  |  {p['cor'].upper()} ({p['classificacao']})  |  "
                f"{p['prioridade_legal']} (Idade: {p['idade']})  |  Entrada: {p['hora_entrada']}\n"
                f"CIAP-2: {p.get('codigo_ciap2', '-')}  |  Descritor: {p.get('discriminador', '-')}\n"
                f"Queixa: {p['queixa_resumida']}"
            )
            ctk.CTkLabel(
                item_frame,
                text=texto_linha,
                font=ctk.CTkFont(size=12),
                text_color="#1E293B",
                justify="left",
                anchor="w",
                wraplength=760
            ).grid(row=0, column=1, sticky="w", pady=8, padx=(0, 10))


if __name__ == "__main__":
    app = SistemaTriagemApp()
    app.mainloop()