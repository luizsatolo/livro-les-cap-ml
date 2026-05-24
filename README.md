# Machine Learning e Avaliação de Impacto em Economia — material suplementar

Repositório oficial dos dados e scripts em R que reproduzem os quatro
exemplos aplicados do capítulo **"Machine Learning e Avaliação de Impacto
em Economia: Fundamentos, Conexões e Aplicações"**, de Ana Lúcia Kassouf
(ESALQ/USP) e Luiz Fernando Satolo (ITA), publicado no livro **(no prelo)**.

O capítulo posiciona o aprendizado de máquina (*Machine Learning*, ML)
como complemento — e não substituto — da econometria causal e ilustra,
com dados públicos brasileiros, como cada estratégia se conecta ao seu
análogo econométrico. Os quatro exemplos foram concebidos para serem
totalmente reprodutíveis a partir dos arquivos deste repositório.

---

## Sumário

1. [Como o repositório está organizado](#como-o-repositório-está-organizado)
2. [Visão geral dos quatro exemplos](#visão-geral-dos-quatro-exemplos)
3. [Requisitos comuns](#requisitos-comuns)
4. [Como reproduzir um exemplo](#como-reproduzir-um-exemplo)
5. [Fontes de dados públicas utilizadas](#fontes-de-dados-públicas-utilizadas)
6. [Como citar](#como-citar)
7. [Licença](#licença)

---

## Como o repositório está organizado

Cada exemplo do capítulo ocupa uma pasta independente, com a mesma
estrutura interna:

```
livro-les-cap-ml/
├── README.md                                  (este arquivo)
├── LICENSE                                    (MIT)
├── .gitignore
├── exemplo1-produtividade-soja/
│   ├── README.md                              (passo a passo do Exemplo 1)
│   ├── R/exemplo1.R
│   ├── dados/base_final.csv
│   └── resultados_modelos/                    (saídas geradas pelo script)
├── exemplo2-ebia-pof/
│   ├── README.md
│   ├── R/exemplo2.R
│   ├── dados/base_ebia.csv
│   └── resultados_modelos/
├── exemplo3-pronaf-dml/
│   ├── README.md
│   ├── R/exemplo3.R
│   ├── dados/base_pronaf.csv
│   └── resultados_modelos/
└── exemplo4-assistencia-tecnica-causal-forest/
    ├── README.md
    ├── R/exemplo4.R
    ├── dados/base_assistencia_tecnica.csv
    └── resultados_modelos/
```

A pasta `resultados_modelos/` de cada exemplo é mantida vazia no
repositório (somente com um `.gitkeep`) e populada quando o script é
executado localmente. A documentação detalhada de cada exemplo — modelos
comparados, dicionário de variáveis, dependências, saídas esperadas e
solução de problemas — está no `README.md` da respectiva subpasta.

---

## Visão geral dos quatro exemplos

| # | Pasta | Seção no capítulo | Método de ML | Pergunta empírica |
|---|---|---|---|---|
| 1 | [`exemplo1-produtividade-soja/`](exemplo1-produtividade-soja/) | 4.3.1 — Regularização (LASSO, Ridge, Elastic Net) | LASSO + efeitos fixos dinâmicos + modelo híbrido | Quão bem é possível **prever** a produtividade municipal da soja no Brasil (2004–2021) com clima, solo e indicadores socioeconômicos? |
| 2 | [`exemplo2-ebia-pof/`](exemplo2-ebia-pof/) | 4.3.2 — Árvores, Random Forest e Gradient Boosting | Random Forest vs. logística multinomial | Quais domicílios da POF 2017–2018 apresentam maior probabilidade de **insegurança alimentar** (EBIA) em cada nível de gravidade? |
| 3 | [`exemplo3-pronaf-dml/`](exemplo3-pronaf-dml/) | 5.2 — Double/Debiased Machine Learning | DML com LASSO e Random Forest como aprendizes | Qual o **efeito causal** do crédito do PRONAF sobre a produtividade municipal do milho (2013–2021)? |
| 4 | [`exemplo4-assistencia-tecnica-causal-forest/`](exemplo4-assistencia-tecnica-causal-forest/) | 5.3 — Causal Forest e efeitos heterogêneos | Causal Forest (grf) com peso de sobreposição | Para **quem** o efeito da assistência técnica agrícola é maior? CATE municipal em cross-section 2017. |

Cada exemplo é autocontido: pode ser executado de forma independente
sem depender dos demais.

---

## Requisitos comuns

- **R** ≥ 4.2
- Pacotes utilizados (variam por exemplo, veja o `README.md` de cada
  pasta):
  `dplyr`, `tidyr`, `readr`, `glmnet`, `fixest`, `ranger`,
  `nnet`, `caret`, `grf`, `DoubleML`, `mlr3`, `ggplot2`,
  `yardstick`, `here`, `vroom`.

Instalação rápida do superconjunto de pacotes:

```r
install.packages(c(
  "dplyr", "tidyr", "readr", "vroom",
  "glmnet", "fixest", "ranger", "nnet", "caret",
  "grf", "DoubleML", "mlr3", "mlr3learners", "mlr3tuning",
  "ggplot2", "yardstick", "here"
))
```

---

## Como reproduzir um exemplo

1. Clone (ou faça download) deste repositório:

   ```bash
   git clone https://github.com/luizsatolo/livro-les-cap-ml.git
   cd livro-les-cap-ml
   ```

2. Entre na pasta do exemplo desejado:

   ```bash
   cd exemplo1-produtividade-soja      # ou exemplo2-..., exemplo3-..., exemplo4-...
   ```

3. Execute o script em R (a partir da pasta do exemplo, para que os
   caminhos relativos `dados/...` e `resultados_modelos/...`
   funcionem):

   ```bash
   Rscript R/exemplo1.R
   ```

   As tabelas, métricas e figuras serão gravadas em
   `resultados_modelos/`. Consulte o `README.md` da pasta para
   informações detalhadas sobre cada saída.

---

## Fontes de dados públicas utilizadas

Os arquivos `dados/*.csv` deste repositório foram montados pelos autores
a partir de fontes públicas brasileiras e internacionais:

- **IBGE** — Produção Agrícola Municipal (PAM), Censo Agropecuário 2017,
  Pesquisa de Orçamentos Familiares (POF) 2017–2018, Estimativas de
  População.
- **ERA5** (ECMWF/Copernicus) — variáveis climáticas reanalisadas
  (temperatura, precipitação, índices de extremos).
- **MapBiomas** — atributos edáficos e de uso do solo.
- **SICOR/Banco Central** — contratações de crédito rural,
  incluindo PRONAF.

Os respectivos `README.md` por exemplo documentam o dicionário de
variáveis e o tratamento aplicado a cada base.

---

## Como citar

> Kassouf, A. L., & Satolo, L. F. (2026). *Machine Learning e Avaliação
> de Impacto em Economia: Fundamentos, Conexões e Aplicações*. In *[no
> prelo]*.

Para citar o repositório de código e dados:

> Kassouf, A. L., & Satolo, L. F. (2026). *livro-les-cap-ml: dados e
> scripts dos exemplos do capítulo Machine Learning e Avaliação de
> Impacto em Economia* (versão 1.0). Disponível em:
> https://github.com/luizsatolo/livro-les-cap-ml.

---

## Licença

Distribuído sob a [Licença MIT](LICENSE). O usuário é livre para usar,
modificar e redistribuir o código e as bases consolidadas, mantendo a
atribuição aos autores e às fontes públicas originais.
