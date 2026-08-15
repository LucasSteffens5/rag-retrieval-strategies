# RAG — Estratégias de Recuperação

Este repositório prepara um corpus documental de concurso docente e executa um experimento comparativo de estratégias de recuperação em sistemas RAG. O fluxo Docker inclui preparação do corpus, ingestão vetorial em Qdrant, execução do benchmark, métricas locais e avaliação RAGAS opcional. O objetivo é manter uma execução reprodutível sem depender de ambiente Python local.

## Pré-requisitos

- Docker e Docker Compose.
- GPU NVIDIA disponível para os contêineres, com suporte ao NVIDIA Container Toolkit ou Docker Desktop com GPU habilitada.
- Acesso de rede na primeira execução para baixar modelos do Ollama e do Hugging Face.
- `GOOGLE_API_KEY` apenas quando a avaliação RAGAS for executada.

## Execução Ponta a Ponta

Execute os comandos no PowerShell a partir da raiz do repositório.

```powershell
cd .\experiment\rag_retrieval_strategies\docker
docker compose build rag-pipeline
docker compose up -d ollama qdrant
```

Defina um identificador para a rodada. O identificador cria uma pasta isolada em `experiment\rag_retrieval_strategies\results\runs\`.

```powershell
$env:EXPERIMENT_RUN_ID = "rodada_001"
```

Para executar sem RAGAS:

```powershell
$env:RUN_RAGAS = "0"
docker compose run --rm rag-pipeline python /app/experiment/rag_retrieval_strategies/src/evaluation/run_full_experiment.py
```

Para executar com RAGAS:

```powershell
$env:RUN_RAGAS = "1"
$env:GOOGLE_API_KEY = "<sua-chave-google>"
docker compose run --rm rag-pipeline python /app/experiment/rag_retrieval_strategies/src/evaluation/run_full_experiment.py
```

Se a rodada já existir e precisar ser substituída, use:

```powershell
$env:FORCE_CLEAN_RUN = "1"
docker compose run --rm rag-pipeline python /app/experiment/rag_retrieval_strategies/src/evaluation/run_full_experiment.py
```

## Saídas

Para `EXPERIMENT_RUN_ID=rodada_001`, os resultados ficam em:

```text
experiment\rag_retrieval_strategies\results\runs\rodada_001\
```

Principais subdiretórios:

| Diretório | Conteúdo |
|---|---|
| `raw\` | JSONs brutos do benchmark por estratégia, modelo e repetição. |
| `metrics\retrieval\` | Métricas locais de recuperação, incluindo Recall@5 e nDCG@5. |
| `metrics\latency\` | Estatísticas de latência por etapa da pipeline. |
| `analysis\scope_behavior\` | Análise complementar de perguntas sem documentos de referência. |
| `ragas\` | Sidecars RAGAS, quando `RUN_RAGAS=1` e `GOOGLE_API_KEY` está definida. |
| `final_results\` | Artefatos consolidados, gerados separadamente após a avaliação RAGAS: 4 PNGs e 3 tabelas CSV. |

## Scripts Individuais no Docker

O orquestrador `run_full_experiment.py` é o caminho recomendado. Os comandos abaixo são úteis para depuração ou reexecução controlada de etapas específicas.

Preparar corpus:

```powershell
docker compose run --rm --no-deps rag-pipeline python /app/database/prepare_corpus.py
```

Executar ingestão vetorial, após `qdrant` estar ativo:

```powershell
docker compose run --rm --no-deps rag-pipeline python /app/experiment/rag_retrieval_strategies/src/ingest.py
```

Executar benchmark em uma pasta de rodada específica:

```powershell
docker compose run --rm -e RESULTS_DIR=/app/experiment/rag_retrieval_strategies/results/runs/rodada_001 rag-pipeline python /app/experiment/rag_retrieval_strategies/src/evaluation/run_benchmark.py
```

Calcular métricas locais:

```powershell
docker compose run --rm -e RESULTS_DIR=/app/experiment/rag_retrieval_strategies/results/runs/rodada_001 rag-pipeline python /app/experiment/rag_retrieval_strategies/src/evaluation/run_metrics.py
```

Executar RAGAS:

```powershell
$env:GOOGLE_API_KEY = "<sua-chave-google>"
docker compose run --rm -e RESULTS_DIR=/app/experiment/rag_retrieval_strategies/results/runs/rodada_001 rag-pipeline python /app/experiment/rag_retrieval_strategies/src/evaluation/ragas_evaluator.py
```

Inspecionar avaliações RAGAS com scores ausentes, sem reprocessar:

```powershell
docker compose run --rm -e RESULTS_DIR=/app/experiment/rag_retrieval_strategies/results/runs/rodada_001 rag-pipeline python /app/experiment/rag_retrieval_strategies/src/evaluation/retry_ragas_failed.py --dry-run
```

Reprocessar avaliações RAGAS com scores ausentes:

```powershell
$env:GOOGLE_API_KEY = "<sua-chave-google>"
docker compose run --rm -e RESULTS_DIR=/app/experiment/rag_retrieval_strategies/results/runs/rodada_001 rag-pipeline python /app/experiment/rag_retrieval_strategies/src/evaluation/retry_ragas_failed.py
```

Gerar artefatos finais para o artigo, após a avaliação RAGAS:

```powershell
docker compose run --rm rag-pipeline python /app/experiment/rag_retrieval_strategies/analysis/generate_final_results.py --run-root /app/experiment/rag_retrieval_strategies/results/runs/rodada_001
```

O orquestrador `run_full_experiment.py` não gera `final_results` automaticamente. Essa etapa requer os 45 sidecars RAGAS da rodada e deve ser executada separadamente.

## Verificação Rápida do Ambiente

Validar o Compose:

```powershell
docker compose config
```

Validar sintaxe dos módulos dentro da imagem:

```powershell
docker compose run --rm --no-deps rag-pipeline python -m compileall -q /app/database /app/experiment/rag_retrieval_strategies/src /app/experiment/rag_retrieval_strategies/analysis
```

Validar preparação do corpus dentro do contêiner:

```powershell
docker compose run --rm --no-deps rag-pipeline python /app/database/prepare_corpus.py
```

## Observações Operacionais

- Este repositório foi testado exclusivamente em sistemas operacionais Windows; a execução em outros sistemas não foi validada.
- A primeira execução pode ser longa porque baixa LLMs do Ollama, o embedding BGE-M3 e o reranker.
- A matriz completa atual executa 5 estratégias × 3 LLMs × 1 embedding × 3 repetições × 50 perguntas.
- `FORCE_CLEAN_RUN=1` remove a pasta da rodada indicada por `EXPERIMENT_RUN_ID`. Na execução completa, o conteúdo de `corpus_processado` também é esvaziado e recriado; os volumes Docker não são removidos.
- Para encerrar os serviços auxiliares, execute `docker compose down`. Para remover todos os volumes declarados — Ollama, Qdrant, cache do Hugging Face e corpus processado — use `docker compose down -v`.
