import os
import re
import fitz  # pymupdf
import ollama
import chromadb

PASTA_DOCS = "documentos_rag"
PASTA_BANCO = "banco_rag_saude"
MODELO_EMBED = "bge-m3"

# Recorte inteligente: indexa apenas as páginas de valor clínico/técnico de cada PDF
CONFIG_DOCUMENTOS = [
    {
        "arquivo": "classificação de atendimento.pdf",
        "fonte": "Protocolo_UPA",
        "pag_inicio": 10,   # A partir de "Como acontece o ACCR", cores, 24 fluxogramas e Anexos
        "pag_fim": 39,
        "chunk_size": 950,
        "overlap": 150
    },
    {
        "arquivo": "CIAP Brasil_atualizado.pdf",
        "fonte": "CIAP2_Brasil",
        "pag_inicio": 72,   # Capítulo 10: Lista Tabular completa (Componentes 1 e 7 + CID-10)
        "pag_fim": 165,
        "chunk_size": 850,
        "overlap": 120
    },
    {
        "arquivo": "caderno_28.pdf",
        "fonte": "Caderno28_MS",
        "pag_inicio": 14,   # Introdução de Risco/Vulnerabilidade + 28 capítulos de queixas comuns
        "pag_fim": 258,
        "chunk_size": 1050,
        "overlap": 180
    }
]


def limpar_texto_pdf(texto):
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


def dividir_em_chunks(texto, tamanho, overlap):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + tamanho
        trecho = texto[inicio:fim].strip()
        if len(trecho) > 100:
            chunks.append(trecho)
        inicio += tamanho - overlap
    return chunks


def vetorizar_todos_documentos():
    print(f"[*] Inicializando banco vetorial local em './{PASTA_BANCO}'...")
    cliente = chromadb.PersistentClient(path=PASTA_BANCO)

    # Recria a coleção do zero para evitar duplicatas se rodar novamente
    try:
        cliente.delete_collection("protocolos_urgencia")
    except Exception:
        pass
    colecao = cliente.create_collection(name="protocolos_urgencia")

    total_global = 0

    for cfg in CONFIG_DOCUMENTOS:
        caminho = os.path.join(PASTA_DOCS, cfg["arquivo"])
        if not os.path.exists(caminho):
            print(f"[!] AVISO: Arquivo não encontrado: {caminho} (pulando...)")
            continue

        print(f"\n[*] Lendo '{cfg['arquivo']}' (Fonte: {cfg['fonte']})...")
        doc = fitz.open(caminho)
        total_pags = len(doc)
        p_ini = max(0, cfg["pag_inicio"])
        p_fim = min(total_pags, cfg["pag_fim"])

        textos_lote = []
        metadados_lote = []
        ids_lote = []

        for num_pag in range(p_ini, p_fim):
            texto_pag = limpar_texto_pdf(doc[num_pag].get_text("text"))
            if len(texto_pag) < 80:
                continue

            chunks = dividir_em_chunks(texto_pag, cfg["chunk_size"], cfg["overlap"])
            for idx_c, chunk in enumerate(chunks):
                total_global += 1
                textos_lote.append(chunk)
                metadados_lote.append({
                    "fonte": cfg["fonte"],
                    "arquivo": cfg["arquivo"],
                    "pagina": num_pag + 1
                })
                ids_lote.append(f"{cfg['fonte']}_p{num_pag+1}_c{idx_c}_{total_global}")

        doc.close()
        print(f"    -> {len(textos_lote)} fragmentos gerados. Calculando embeddings com '{MODELO_EMBED}'...")

        # Processa em lotes de 24 fragmentos para velocidade máxima na GPU
        tamanho_lote = 24
        for i in range(0, len(textos_lote), tamanho_lote):
            fatia_txt = textos_lote[i:i + tamanho_lote]
            fatia_meta = metadados_lote[i:i + tamanho_lote]
            fatia_ids = ids_lote[i:i + tamanho_lote]

            resp = ollama.embed(model=MODELO_EMBED, input=fatia_txt)
            vetores = resp["embeddings"]

            colecao.add(
                ids=fatia_ids,
                embeddings=vetores,
                documents=fatia_txt,
                metadatas=fatia_meta
            )
            print(f"       Progresso {cfg['fonte']}: {min(i + tamanho_lote, len(textos_lote))}/{len(textos_lote)}", end="\r")

        print(f"\n    [V] '{cfg['fonte']}' vetorizado com sucesso!")

    # Libera a VRAM do modelo de embeddings ao terminar
    try:
        ollama.generate(model=MODELO_EMBED, prompt="", keep_alive=0)
    except Exception:
        pass

    print(f"\n[+] CONCLUÍDO! {total_global} blocos clínicos indexados em './{PASTA_BANCO}'.")


if __name__ == "__main__":
    vetorizar_todos_documentos()