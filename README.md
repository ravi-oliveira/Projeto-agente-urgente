# Sistema Inteligente de Triagem e Acolhimento para Pronto-Socorro

Proposta de Inteligência Artificial local aplicada à gestão de fluxo, acolhimento e classificação de risco em unidades de Pronto-Socorro (UPAs e emergências hospitalares). O sistema combina um agente conversacional local (**Qwen 2.5 14B**), busca semântica em protocolos oficiais de saúde (**RAG com ChromaDB + BGE-M3**) e um **motor determinístico de ordenação de fila**.

## Arquitetura do Sistema

O projeto resolve limitações comuns de modelos de linguagem em saúde por meio de uma arquitetura híbrida em 4 camadas:

1. **Triagem Conversacional Guiada (`qwen2.5:14b` via Ollama):** Conduz uma anamnese empática e estruturada, coletando identificação, condições especiais (gestante/PcD), idade, gênero (omitido automaticamente quando gestante), queixa principal, intensidade pela **Escala Verbal Numérica (0–10)**, sintomas correlatos, biometria e fatores de risco. Utiliza raciocínio interno oculto (`pensamento_interno`) antes de cada resposta e interrompe o questionário imediatamente em sinais de risco iminente de morte.
2. **RAG Clínico Local (`ChromaDB` + `bge-m3`):** A base vetorial (`banco_rag_saude/`) já acompanha o repositório e indexa três referenciais oficiais consultados em tempo real durante o atendimento:
   - **Protocolo de Acolhimento e Classificação de Risco (UPA Maringá 2023):** Descritores objetivos de cor e sinais vitais (Protocolo de Manchester).
   - **Caderno de Atenção Básica nº 28 — Vol. II (Ministério da Saúde):** Fluxogramas de demanda espontânea, sinais de alerta e vulnerabilidades.
   - **CIAP-2 Brasil (WONCA / SBMFC):** Codificação padronizada do motivo da consulta.
3. **Motor Determinístico de Prioridades (`GerenciadorFila`):** Garante a segurança clínica na ordenação da fila com chave composta:
   $$\text{Ordem} = (\text{Gravidade Clínica}, \text{Prioridade Legal}, \text{Horário de Chegada})$$
   - **Gravidade Clínica (Protocolo de Manchester):** Vermelho (0 min) > Laranja (10 min) > Amarelo (60 min) > Verde (120 min) > Azul (240 min).
   - **Prioridade Legal (critério de desempate dentro da mesma cor):** Gestante > Idoso (60+) > Criança (<12) > PcD > Normal, com travas determinísticas de idade e gênero em Python.
4. **Interface Clínica Limpa (`CustomTkinter`):** Painel desktop dividido entre a tabela oficial de cores de risco e a área de acolhimento, além de uma janela de monitoramento (`Fila Ordenada`) para demonstração e auditoria em `fila_triagem.json`.

## Pré-requisitos

- **Ollama** instalado e em execução ([ollama.com](https://ollama.com)).
- Modelos locais baixados no Ollama:
  ```bash
  ollama pull qwen2.5:14b
  ollama pull bge-m3