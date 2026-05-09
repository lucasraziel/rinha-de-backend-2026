# Rinha de Backend 2026 — Lucas Rego (`lucasraziel`)

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026): API HTTP que detecta fraude em transações por busca vetorial KNN k=5 sobre 3 milhões de vetores rotulados, dentro de um orçamento de 1 CPU + 350 MB RAM.

## Stack

- **Rust** (stable), `hyper` 1.x sobre `tokio` current-thread.
- **Nginx** alpine como load balancer round-robin.
- Distroless `cc-debian12` como imagem final.
- Build alvo `x86-64-v3` (Haswell+: AVX2 + BMI2 + F16C) — máquina avaliadora é um Mac Mini Late 2014.

## Layout

```
crates/
  api/      Servidor HTTP (rotas /ready e /fraud-score)
  prep/     Pipeline offline: references.json.gz → references.bin
docker/     Dockerfile + nginx.conf
resources/  Constantes do desafio (normalization, mcc_risk) — references.json.gz NÃO entra no git
data/       references.bin gerado pelo prep (NÃO entra no git)
bench/      Cópia do k6 oficial + script local
scripts/    Setup (fetch-resources, prepare-dataset, submit)
```

## Como rodar local

```bash
# 1. Baixar o dataset original
./scripts/fetch-resources.sh

# 2. (F2+) Gerar references.bin
cargo run --release -p prep

# 3. Subir o stack (nginx + 2 réplicas da api) com limites cgroup
docker compose up --build -d

# 4. Sanity check
curl -fsS http://localhost:9999/ready
curl -fsS -X POST http://localhost:9999/fraud-score \
    -H 'content-type: application/json' \
    --data @resources/example-payloads.json | head

# 5. Rodar o teste oficial (precisa do k6 instalado)
./bench/run-local.sh
```

## Distribuição de recursos (1 CPU + 350 MB)

| Container | CPU | Memória |
|---|---:|---:|
| nginx | 0.10 | 25 MB |
| api1 | 0.45 | 160 MB |
| api2 | 0.45 | 160 MB |
| **total** | **1.00** | **345 MB** |

## Roadmap de fases

1. **F1** — esqueleto: handlers stub passando `/ready` e `/fraud-score` com resposta fixa. *(atual)*
2. **F2** — pré-processamento + brute-force escalar exato com 0% de erro de detecção.
3. **F3** — SIMD AVX2 + dataset em f16 padded a 16 dims + zero-alloc no hot path.
4. **F4** — IVF (`nlist=1024`, `nprobe=16-32`) + refine exato top-50.
5. **F5** — branch `submission` + tag imutável no Docker Hub + PR no repo oficial.

## Score-alvo

| Item | Realista | Stretch |
|---|---:|---:|
| `p99` | 3–5 ms | < 2 ms |
| Detection error | < 1 % | < 0.5 % |
| Score total | +4500…+5500 | +5800 |

## Anti-trapaça

- **NUNCA** usar `test/test-data.json` como fonte de referência. Para validar recall do IVF, usar holdout de 1 % do próprio `references.json.gz`.
- Load balancer puro round-robin, sem inspeção/transformação de payload.
- Imagens públicas, MIT license, repo aberto.

## Licença

MIT — ver [LICENSE](./LICENSE).
