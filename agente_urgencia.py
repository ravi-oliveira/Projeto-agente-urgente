import ollama
import json
import os
import datetime
import threading
import customtkinter as ctk

# --- CONFIGURAÇÕES DO SISTEMA ---
MODELO_LLM = "qwen2.5:7b"
ARQUIVO_BANCO = "fila_triagem.json"

# Tabela Oficial de Classificação de Risco (Protocolo de Manchester)
PROTOCOLO_MANCHESTER = [
    {
        "nivel": 1,
        "sigla": "VM",
        "cor_nome": "Vermelho",
        "classificacao": "Emergência",
        "tempo": "Imediato (0 min)",
        "exemplos": "Risco iminente de morte, parada cardiorrespiratória, falta de ar grave ou choque.",
        "hex": "#DC2626",
        "hex_suave": "#FEF2F2",
        "texto_badge": "#FFFFFF"
    },
    {
        "nivel": 2,
        "sigla": "LR",
        "cor_nome": "Laranja",
        "classificacao": "Muito urgente",
        "tempo": "Até 10 minutos (varia até 15–20 min)",
        "exemplos": "Risco grave, dor intensa, sangramento importante ou piora rápida de sinais vitais.",
        "hex": "#EA580C",
        "hex_suave": "#FFF7ED",
        "texto_badge": "#FFFFFF"
    },
    {
        "nivel": 3,
        "sigla": "AM",
        "cor_nome": "Amarelo",
        "classificacao": "Urgente",
        "tempo": "Até 60 minutos",
        "exemplos": "Risco moderado, dor forte controlável, crise hipertensiva com sintomas ou suspeita de fratura.",
        "hex": "#CA8A04",
        "hex_suave": "#FEFCE8",
        "texto_badge": "#FFFFFF"
    },
    {
        "nivel": 4,
        "sigla": "VD",
        "cor_nome": "Verde",
        "classificacao": "Pouco urgente",
        "tempo": "Até 120 minutos",
        "exemplos": "Baixa gravidade, sintomas leves/estáveis, como dor de cabeça leve ou infecção sem sinais de alarme.",
        "hex": "#16A34A",
        "hex_suave": "#F0FDF4",
        "texto_badge": "#FFFFFF"
    },
    {
        "nivel": 5,
        "sigla": "AZ",
        "cor_nome": "Azul",
        "classificacao": "Não urgente",
        "tempo": "Até 240 minutos (ou encaminhamento à UBS)",
        "exemplos": "Quadros crônicos sem agudização, queixas antigas ou demandas eletivas, como troca de receitas.",
        "hex": "#2563EB",
        "hex_suave": "#EFF6FF",
        "texto_badge": "#FFFFFF"
    }
]

MAPA_CORES = {item["cor_nome"].upper(): item for item in PROTOCOLO_MANCHESTER}

# Pesos de Prioridade Legal (Desempate dentro da mesma cor clínica)
PESOS_PRIORIDADE = {
    "GESTANTE": (1, "G", "Gestante"),
    "IDOSO": (2, "I", "Idoso (60+)"),
    "CRIANCA": (3, "C", "Criança / Lactente"),
    "PCD": (4, "P", "Pessoa com Deficiência"),
    "NORMAL": (5, "N", "Atendimento Normal")
}


# --- CAMADA DE DADOS E MOTOR DE ORDENAÇÃO DA FILA ---

class GerenciadorFila:
    def __init__(self, caminho_arquivo=ARQUIVO_BANCO):
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
            "queixa_resumida": dados_triagem.get("queixa_resumida", ""),
            "justificativa_clinica": dados_triagem.get("justificativa_clinica", "")
        }

        self.pacientes.append(registro)
        self._salvar()
        return registro

    def obter_fila_ordenada(self):
        """
        Ordenação determinística:
        1º Gravidade Clínica (1=Vermelho ... 5=Azul)
        2º Prioridade Legal (1=Gestante, 2=Idoso, 3=Criança, 4=PcD, 5=Normal)
        3º Ordem cronológica de chegada (timestamp)
        """
        return sorted(
            self.pacientes,
            key=lambda p: (p["nivel_clinico"], p["peso_prioridade"], p["timestamp"])
        )

    def limpar_fila(self):
        self.pacientes = []
        self._salvar()


# --- MOTOR CONVERSACIONAL DE TRIAGEM (OLLAMA) ---

SYSTEM_PROMPT_TRIAGEM = """
Você é um atendente virtual de acolhimento e triagem de um Pronto-Socorro.
Seu papel é conversar de forma empática, clara, breve e objetiva com o paciente (ou acompanhante) para coletar informações suficientes e classificá-lo segundo o Protocolo de Manchester e prioridades legais.

PROTOCOLO DE MANCHESTER (CORES):
- VERMELHO (Emergência - 0 min): Risco iminente de morte, parada cardiorrespiratória, falta de ar grave, dor no peito intensa com suor frio/irradiação, inconsciência ou choque.
- LARANJA (Muito urgente - até 10 min): Risco grave, dor intensa, sangramento importante, febre muito alta com prostração ou piora rápida de sinais vitais.
- AMARELO (Urgente - até 60 min): Risco moderado, dor forte controlável, crise hipertensiva com sintomas, vômitos persistentes ou suspeita de fratura.
- VERDE (Pouco urgente - até 120 min): Baixa gravidade, sintomas leves/estáveis, dor de cabeça leve, resfriado ou infecção sem sinais de alarme.
- AZUL (Não urgente - até 240 min): Quadros crônicos sem agudização, queixas antigas ou demandas eletivas como troca de receitas e atestados.

PRIORIDADES LEGAIS:
- GESTANTE, IDOSO (60 anos ou mais), CRIANCA (menor de 12 anos), PCD (Pessoa com Deficiência) ou NORMAL.

REGRAS DE CONDUÇÃO:
1. Faça no máximo 1 ou 2 perguntas curtas por vez.
2. Você precisa descobrir: (a) qual o sintoma principal, intensidade e há quanto tempo começou; (b) idade do paciente e se se enquadra em alguma prioridade legal (gestante, idoso, criança, PcD).
3. REGRA DE EMERGÊNCIA IMEDIATA: Se o usuário relatar sinais claros de VERMELHO (ex: falta de ar grave, desmaio, dor forte no peito, hemorragia maciça), NÃO faça mais perguntas — conclua imediatamente a triagem como VERMELHO.
4. Assim que tiver os sintomas e o perfil (ou em caso de emergência imediata), mude "status" para "CONCLUIDO" e preencha "dados_triagem". Enquanto ainda faltar informação, mantenha "status" como "COLETANDO".

RESPONDA SEMPRE E APENAS EM JSON VÁLIDO NO FORMATO EXATO ABAIXO:
{
  "status": "COLETANDO",
  "mensagem_paciente": "Texto educado e direto para o paciente aqui.",
  "dados_triagem": {
    "cor": "VERDE",
    "prioridade_legal": "NORMAL",
    "idade": "Não informada",
    "queixa_resumida": "",
    "justificativa_clinica": ""
  }
}
"""


# --- INTERFACE GRÁFICA CLÍNICA E LIMPA ---

class SistemaTriagemApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("light")
        self.title("Acolhimento e Classificação de Risco — Pronto-Socorro")
        self.geometry("1240x760")
        self.minsize(1050, 650)
        self.configure(fg_color="#F8FAFC")

        self.gerenciador = GerenciadorFila()
        self.historico_llm = []
        self.processando = False
        self.atendimento_concluido = False

        self.grid_columnconfigure(0, weight=5)  # Painel Esquerdo: Tabela de Cores
        self.grid_columnconfigure(1, weight=6)  # Painel Direito: Atendimento IA
        self.grid_rowconfigure(1, weight=1)

        self._construir_cabecalho()
        self._construir_painel_tabela()
        self._construir_painel_atendimento()
        self._iniciar_novo_atendimento()

    def _construir_cabecalho(self):
        header = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=0, height=64, border_width=1, border_color="#E2E8F0")
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        lbl_titulo = ctk.CTkLabel(
            header,
            text="  ✚  PRONTO-SOCORRO  |  TRIAGEM E CLASSIFICAÇÃO DE RISCO",
            font=ctk.CTkFont(family="Ubuntu", size=17, weight="bold"),
            text_color="#0F172A",
            anchor="w"
        )
        lbl_titulo.grid(row=0, column=0, padx=20, pady=16, sticky="w")

        btn_fila = ctk.CTkButton(
            header,
            text="Fila Ordenada (Pesquisa)",
            width=185,
            height=34,
            corner_radius=8,
            fg_color="#F1F5F9",
            hover_color="#E2E8F0",
            text_color="#334155",
            border_width=1,
            border_color="#CBD5E1",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._abrir_janela_fila
        )
        btn_fila.grid(row=0, column=1, padx=20, pady=14, sticky="e")

    def _construir_painel_tabela(self):
        """Exibe apenas a tabela limpa de classificação de risco igual às unidades de saúde."""
        container = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=14, border_width=1, border_color="#E2E8F0")
        container.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=20)
        container.grid_columnconfigure(0, weight=1)

        lbl_cab = ctk.CTkLabel(
            container,
            text="CLASSIFICAÇÃO DE RISCO (PROTOCOLO DE MANCHESTER)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#0F172A",
            anchor="w"
        )
        lbl_cab.pack(fill="x", padx=20, pady=(18, 4))

        lbl_sub = ctk.CTkLabel(
            container,
            text="O atendimento é organizado pela gravidade clínica. Prioridades legais (gestantes, idosos, crianças e PcD) têm preferência dentro da mesma cor.",
            font=ctk.CTkFont(size=11),
            text_color="#64748B",
            wraplength=480,
            justify="left",
            anchor="w"
        )
        lbl_sub.pack(fill="x", padx=20, pady=(0, 14))

        # Cards das 5 cores oficiais
        for item in PROTOCOLO_MANCHESTER:
            linha = ctk.CTkFrame(
                container,
                fg_color=item["hex_suave"],
                corner_radius=10,
                border_width=1,
                border_color="#E2E8F0"
            )
            linha.pack(fill="x", padx=18, pady=5)

            # Faixa lateral colorida
            faixa = ctk.CTkFrame(linha, fg_color=item["hex"], width=10, corner_radius=6)
            faixa.pack(side="left", fill="y", padx=(6, 12), pady=6)

            conteudo = ctk.CTkFrame(linha, fg_color="transparent")
            conteudo.pack(side="left", fill="both", expand=True, pady=8, padx=(0, 12))

            topo = ctk.CTkFrame(conteudo, fg_color="transparent")
            topo.pack(fill="x")

            lbl_cor = ctk.CTkLabel(
                topo,
                text=f"{item['cor_nome'].upper()} — {item['classificacao']}",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=item["hex"]
            )
            lbl_cor.pack(side="left")

            lbl_tempo = ctk.CTkLabel(
                topo,
                text=item["tempo"],
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#334155"
            )
            lbl_tempo.pack(side="right")

            lbl_ex = ctk.CTkLabel(
                conteudo,
                text=item["exemplos"],
                font=ctk.CTkFont(size=12),
                text_color="#475569",
                wraplength=440,
                justify="left",
                anchor="w"
            )
            lbl_ex.pack(fill="x", pady=(3, 0))

    def _construir_painel_atendimento(self):
        container = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=14, border_width=1, border_color="#E2E8F0")
        container.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=20)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)

        # Topo do chat
        topo_chat = ctk.CTkFrame(container, fg_color="transparent")
        topo_chat.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 8))

        lbl_atend = ctk.CTkLabel(
            topo_chat,
            text="ATENDIMENTO DE TRIAGEM",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#0F172A"
        )
        lbl_atend.pack(side="left")

        self.btn_novo = ctk.CTkButton(
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
        )
        self.btn_novo.pack(side="right")

        # Área de mensagens
        self.chat_scroll = ctk.CTkScrollableFrame(
            container,
            fg_color="#F8FAFC",
            corner_radius=10,
            border_width=1,
            border_color="#E2E8F0"
        )
        self.chat_scroll.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 10))
        self.chat_scroll.grid_columnconfigure(0, weight=1)

        # Barra de status + input
        rodape = ctk.CTkFrame(container, fg_color="transparent")
        rodape.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 16))
        rodape.grid_columnconfigure(0, weight=1)

        self.lbl_status = ctk.CTkLabel(
            rodape,
            text="Aguardando resposta do paciente...",
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

    # --- CONTROLE DE CONVERSA E INTERFACE ---

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
        self.entrada_msg.configure(state="normal")
        self.btn_enviar.configure(state="normal")
        self.lbl_status.configure(text="Pronto para acolhimento.")

        self.historico_llm = [{"role": "system", "content": SYSTEM_PROMPT_TRIAGEM}]
        saudacao = (
            "Olá! Bem-vindo ao acolhimento do Pronto-Socorro.\n"
            "Para agilizarmos sua senha de atendimento, por favor me informe: "
            "o que você está sentindo hoje e qual a sua idade?"
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
            f"Tempo Meta de Atendimento: {registro['tempo_meta']}\n"
            f"Prioridade Legal: {registro['prioridade_legal']}  |  Idade: {registro['idade']}\n"
            f"Resumo Clínico: {registro['queixa_resumida']}\n"
            f"Critério da Triagem: {registro['justificativa_clinica']}\n"
            f"Horário de Registro: {registro['hora_entrada']} ({registro['data']})"
        )
        ctk.CTkLabel(
            card,
            text=detalhes,
            font=ctk.CTkFont(size=12),
            text_color="#1E293B",
            wraplength=500,
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
        self.historico_llm.append({"role": "user", "content": texto})

        self.processando = True
        self.btn_enviar.configure(state="disabled")
        self.lbl_status.configure(text="Analisando relato clínico com IA local...")

        threading.Thread(target=self._processar_triagem_ia, daemon=True).start()

    def _processar_triagem_ia(self):
        try:
            resp = ollama.chat(
                model=MODELO_LLM,
                messages=self.historico_llm,
                format="json"
            )
            conteudo_bruto = resp["message"]["content"]
            self.historico_llm.append({"role": "assistant", "content": conteudo_bruto})

            dados = json.loads(conteudo_bruto)
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

    # --- JANELA SECUNDÁRIA PARA PESQUISADORES (FILA ORDENADA) ---

    def _abrir_janela_fila(self):
        janela = ctk.CTkToplevel(self)
        janela.title("Monitor de Fila Prioritária — Visão para Pesquisadores")
        janela.geometry("820x520")
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

            ctk.CTkFrame(item_frame, fg_color=cor_hex, width=8, corner_radius=4).pack(side="left", fill="y", padx=(6, 10), pady=6)

            texto_linha = (
                f"{pos}º  |  Senha: {p['senha']}  |  {p['cor'].upper()} ({p['classificacao']})  |  "
                f"{p['prioridade_legal']} (Idade: {p['idade']})  |  Entrada: {p['hora_entrada']}\n"
                f"Queixa: {p['queixa_resumida']}"
            )
            ctk.CTkLabel(
                item_frame,
                text=texto_linha,
                font=ctk.CTkFont(size=12),
                text_color="#1E293B",
                justify="left",
                anchor="w",
                wraplength=730
            ).pack(side="left", fill="x", expand=True, pady=8, padx=(0, 10))


if __name__ == "__main__":
    app = SistemaTriagemApp()
    app.mainloop()