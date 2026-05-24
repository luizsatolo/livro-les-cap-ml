# Exemplo 3 — Efeito do PRONAF sobre a produtividade do milho municipal via Double/Debiased Machine Learning (painel 2013–2021)

Material de apoio do **Exemplo 3** do capítulo *"Machine Learning em
Avaliação de Impacto na Economia"*, ilustrando a seção 5.2
(Double/Debiased Machine Learning). O exemplo estima, em **painel anual
2013–2021**, o efeito causal da intensidade de exposição ao **Programa
Nacional de Fortalecimento da Agricultura Familiar (PRONAF)** sobre a
**produtividade do milho em municípios brasileiros**, controlando por
um vetor rico de características climáticas, edáficas, demográficas e
econômicas, por meio do *Double/Debiased Machine Learning* (DML) de
Chernozhukov et al. (2018).

---

## Sumário

1. [Contexto](#contexto)
2. [Modelo e método](#modelo-e-método)
3. [Estrutura do repositório](#estrutura-do-repositório)
4. [Requisitos](#requisitos)
5. [Como reproduzir o exemplo](#como-reproduzir-o-exemplo)
6. [A base de dados](#a-base-de-dados)
7. [Dicionário de variáveis](#dicionário-de-variáveis)
8. [O que o script faz, passo a passo](#o-que-o-script-faz-passo-a-passo)
9. [Saídas geradas](#saídas-geradas)
10. [Resultados esperados](#resultados-esperados)
11. [Solução de problemas](#solução-de-problemas)
12. [Como citar](#como-citar)
13. [Licença](#licença)

---

## Contexto

Avaliar o efeito causal de políticas públicas em painéis com muitas
covariáveis é um problema clássico em economia aplicada. Em situações
em que o tratamento é observado, mas há um número grande de
características potencialmente confundidoras, métodos econométricos
tradicionais como MQO sofrem com o trade-off entre **viés de variável
omitida** (poucos controles) e **viés de pós-seleção** (seleção
arbitrária de controles).

O **Double/Debiased Machine Learning (DML)**, desenvolvido por
Chernozhukov et al. (2018), oferece uma solução para esse problema. A
ideia é estimar duas funções *nuisance* — `g(X) = E[Y|X]` e
`m(X) = E[D|X]` — por aprendizado de máquina e construir um estimador
"ortogonal" que é √n-consistente e assintoticamente normal sob
condições de regularidade brandas, **mesmo quando X é de alta
dimensão**.

Aqui aplicamos DML em **painel anual 2013–2021** para estimar o efeito
do **valor anual contratado de PRONAF (R$, em log)** sobre a
**produtividade do milho em kg/ha (em log)**, condicional a clima
(ERA5), solo (MapBiomas), estrutura econômica (IBGE), macrorregião e
ano. A inferência usa **erros-padrão robustos a agrupamento (cluster)
por município**, via `DoubleMLClusterData`.

---

## Modelo e método

O exemplo implementa o **modelo parcialmente linear** de Robinson
(1988) com a correção de Chernozhukov et al. (2018), aplicado ao
painel:

```
Y_it = θ · D_it + g(X_it) + u_it        E[u_it | X_it, D_it] = 0
D_it =           m(X_it) + v_it         E[v_it | X_it]        = 0
```

onde:

- `Y_it` = `log(produtividade_milho)` do município *i* no ano *t*
  (kg/ha, PAM/IBGE, em log natural);
- `D_it` = `log(1 + valor_pronaf)` — valor anual contratado de PRONAF
  em R$ (SICOR/Bacen). Municípios sem contratação em um dado ano
  entram com `D = 0`;
- `X_it` = vetor de 29 covariáveis: 6 climáticas (ERA5), 3 edáficas
  (MapBiomas), 8 socioeconômicas (IBGE, log-transformadas onde de
  escala), 4 *dummies* de macrorregião (Norte como referência) e
  8 *dummies* de ano (2013 como referência);
- `θ` = efeito causal de interesse — interpretação semi-elástica:
  variação esperada em log(produtividade) por unidade de aumento em
  `log(1 + PRONAF)`.

### Algoritmo (DML / partialling-out com clustering)

1. Particiona-se a amostra em *K* subconjuntos por *cluster*
   (município), preservando a estrutura de painel
   (cross-fitting, *K* = 5).
2. Em *K-1* subconjuntos, estimam-se `ĝ(X)` e `m̂(X)` por LASSO
   (`cv.glmnet`) ou Random Forest (`ranger`).
3. No subconjunto excluído, calculam-se os resíduos
   `Y - ĝ(X)` e `D - m̂(X)`.
4. Repete-se para cada fold; estima-se θ pela regressão dos resíduos
   *out-of-fold*.
5. O erro-padrão de θ̂ é **cluster-robusto** (clustering por
   `id_municipio`); IC95% via aproximação normal.

O exemplo roda o procedimento duas vezes, alternando o aprendiz
auxiliar entre **LASSO** e **Random Forest**.

> **Caveat de identificação.** O DML elimina o viés de
> pós-regularização ao incluir muitas covariáveis, mas **não resolve
> viés por variável omitida**. Confundidores invariantes no tempo
> (por exemplo, qualidade do extensionismo rural ou capital social
> local) não capturados em `X_it` ainda contaminam θ̂. A inclusão
> das *dummies* de macrorregião e de ano absorve heterogeneidade
> regional e choques temporais comuns, mas não substitui um conjunto
> completo de efeitos fixos de município. Para identificação robusta,
> considerar uma versão *within* (demeaning por município) ou
> estratégias de variáveis instrumentais (seções 3.4 e 5.4 do
> capítulo).

---

## Estrutura do repositório

Esta pasta é parte do repositório consolidado [`livro-les-cap-ml`](https://github.com/luizsatolo/livro-les-cap-ml) e segue a estrutura abaixo dentro de `exemplo3-pronaf-dml/`:

```
exemplo3-pronaf-dml/
├── R/
│   └── exemplo3.R                       # script principal (DML painel)
├── dados/
│   └── base_pronaf.csv                  # painel município × ano 2013-2021
├── resultados_modelos/                  # criada/atualizada na execução
└── README.md
```

---

## Requisitos

- **R** ≥ 4.1 (recomendado 4.3 ou superior).
- Conexão à internet apenas na primeira execução para instalar
  pacotes ausentes do CRAN. **O script não faz nenhuma chamada de
  rede para acessar dados** — toda a base está em `dados/base_pronaf.csv`.

### Pacotes R

O script instala automaticamente qualquer pacote ausente:

| Pacote | Função no exemplo |
|---|---|
| `readr` | leitura de CSV |
| `dplyr` | manipulação de data frames |
| `tidyr` | drop_na |
| `tibble` | tibbles |
| `ggplot2` | gráficos |
| `DoubleML` | implementação do DML (incl. `DoubleMLClusterData`) |
| `mlr3` | framework de ML usado por DoubleML |
| `mlr3learners` | aprendizes (LASSO, Random Forest) |
| `glmnet` | backend do LASSO |
| `ranger` | backend do Random Forest |

Para instalar manualmente, antes de executar:

```r
install.packages(c(
  "readr", "dplyr", "tidyr", "tibble", "ggplot2",
  "DoubleML", "mlr3", "mlr3learners", "glmnet", "ranger"
))
```

---

## Como reproduzir o exemplo

### Opção A — Linha de comando

```bash
git clone https://github.com/luizsatolo/livro-les-cap-ml.git
cd livro-les-cap-ml/exemplo3-pronaf-dml
Rscript R/exemplo3.R dados/base_pronaf.csv
```

### Opção B — Dentro do R / RStudio (recomendada no Windows)

```r
# defina o diretório de trabalho na raiz do repositório
setwd("caminho/para/o/repositorio")

# executa o script
source("R/exemplo3.R")
```

> Se executado sem argumento, o script tenta ler `dados/base_pronaf.csv`
> a partir do diretório de trabalho atual.

A execução típica leva entre 1 e 4 minutos (o Random Forest com 500
árvores em cada fold é o passo mais demorado).

---

## A base de dados

O arquivo único `dados/base_pronaf.csv` é um **painel anual
município × ano** com cerca de **19.800 observações** distribuídas em
**9 anos (2013–2021)** e **~2.556 municípios brasileiros produtores
de milho**. O painel é não-balanceado: cada município entra na base
apenas nos anos em que houve produção de milho reportada à PAM/IBGE.
As fontes originais (consolidadas neste único CSV) são:

- **Produtividade do milho** — IBGE, Produção Agrícola Municipal (PAM);
- **Variáveis climáticas** — [ERA5](https://cds.climate.copernicus.eu/)
  (reanálise ECMWF/Copernicus), com indicadores anuais de temperatura,
  precipitação e extremos por município;
- **Variáveis edáficas** — [MapBiomas](https://brasil.mapbiomas.org/)
  (carbono médio do solo, teores de argila e areia na camada 0–10 cm);
- **Variáveis socioeconômicas** — IBGE (PIB total e setorial,
  população, área e densidade);
- **Valor e número de contratos PRONAF** — SICOR (Sistema de Operações
  de Crédito Rural) do **Banco Central do Brasil**, agregados por
  município e ano.
- **Macrorregião** — derivada do código IBGE do município (primeiro
  dígito).

O período 2013–2021 reflete a disponibilidade do recorte SICOR
utilizado. Municípios sem contratação PRONAF em um dado ano constam na
base com `valor_pronaf = 0` e `contratos_pronaf = 0` (interpretação:
ausência de exposição naquele ano).

---

## Dicionário de variáveis

### Identificadores e estrutura do painel

| Variável | Tipo | Descrição |
|---|---|---|
| `id_municipio` | character | Código IBGE do município (7 dígitos) |
| `ano` | inteiro | Ano de referência (2013–2021) |
| `regiao` | character | Macrorregião do município (Norte, Nordeste, Sudeste, Sul, Centro-Oeste) |

### Variável de resposta (Y) e tratamento (D)

| Variável | Papel | Fonte | Descrição | Unidade |
|---|---|---|---|---|
| `produtividade_milho` | base de Y | IBGE/PAM | Produtividade do milho (total) | kg/ha |
| `valor_pronaf` | base de D | SICOR/Bacen | Valor anual contratado de PRONAF | R$ |
| `contratos_pronaf` | descritor | SICOR/Bacen | Número anual de contratos PRONAF | — |

No script:
- `Y = log(produtividade_milho)`
- `D = log(1 + valor_pronaf)` (`log1p` para suportar zeros).

### Controles climáticos (ERA5)

| Variável | Descrição | Unidade |
|---|---|---|
| `temperatura_media` | Temperatura média anual | °C |
| `temperatura_maxima` | Temperatura máxima anual | °C |
| `dias_quentes` | Dias acima do percentil 95 (TX95p) | dias |
| `precipitacao_anual` | Precipitação acumulada anual | mm |
| `precipitacao_extrema` | Máxima precipitação em 5 dias consecutivos (Rx5day) | mm |
| `dias_secos` | Maior sequência de dias consecutivos secos (CDD) | dias |

### Controles edáficos (MapBiomas)

| Variável | Descrição | Unidade |
|---|---|---|
| `carbono_solo` | Carbono médio no solo | g/kg |
| `teor_argila` | Teor de argila (0–10 cm) | % |
| `teor_areia` | Teor de areia (0–10 cm) | % |

### Controles socioeconômicos (IBGE)

| Variável original | Versão usada no DML |
|---|---|
| `pib_total` | `log_pib_total` |
| `pib_agropecuaria` | `log_pib_agropec` |
| `pib_industria` | `log_pib_industria` |
| `pib_servicos` | `log_pib_servicos` |
| `populacao_total` | `log_populacao` |
| `area_municipio` | `log_area` |
| `pib_percapita` | `log_pib_percapita` |
| `densidade_demografica` | `densidade_demografica` |

> As variáveis socioeconômicas de escala recebem transformação
> `log(max(x, 1))` antes de entrarem no DML.

### Dummies criadas no script

- **Macrorregião** (Norte = referência): `regiao_NE`, `regiao_SE`,
  `regiao_SUL`, `regiao_CO`.
- **Ano** (2013 = referência): `ano_2014`, `ano_2015`, …, `ano_2021`.

Total de covariáveis em X: 6 (clima) + 3 (solo) + 8 (socio) +
4 (região) + 8 (ano) = **29**.

---

## O que o script faz, passo a passo

O script `R/exemplo3.R` está organizado em 13 seções comentadas. Em
resumo:

1. **Carrega pacotes** (instala os ausentes) e fixa `set.seed(123)`.
2. **Lê argumentos** da linha de comando e cria o diretório de saída
   (`resultados_modelos/`).
3. **Lê e valida** a base; aborta se faltar alguma variável obrigatória.
4. **Prepara os dados**: filtra `produtividade_milho > 0` e
   `valor_pronaf ≥ 0`; aplica `log` a Y e às variáveis socioeconômicas
   de escala; aplica `log1p` ao PRONAF; cria *dummies* de macrorregião
   e de ano; cria `id_mun_num` (versão inteira do código IBGE para uso
   como cluster).
5. **Especifica Y, D e X** (29 variáveis em X) e salva
   `base_dml_processada.csv` em `resultados_modelos/`.
6. **Constrói `DoubleMLClusterData`** com `id_mun_num` como variável
   de agrupamento (cluster).
7. **DML com LASSO** (`mlr3::lrn("regr.cv_glmnet")`) como aprendiz
   para `g` e `m`; cross-fitting de 5 folds; score `"partialling out"`.
8. **DML com Random Forest** (`mlr3::lrn("regr.ranger")` com 500
   árvores, `mtry = floor(sqrt(p))`, `min.node.size = 5`).
9. **Tabela comparativa**: θ̂, erro-padrão clustered, IC95%, *z* e
   *p*-valor para cada aprendiz.
10. **Forest plot** dos efeitos estimados (LASSO vs RF).
11. **Resíduos do partialling-out** (Y - ĝ, D - m̂) salvos para
    diagnóstico.
12. **Importância das variáveis** nas equações auxiliares estimadas
    por Random Forest (médias dos *folds*).
13. Mensagem final indicando onde os resultados foram salvos.

---

## Saídas geradas

Todas as saídas são gravadas em `resultados_modelos/`:

| Arquivo | Conteúdo |
|---|---|
| `base_dml_processada.csv` | Base efetivamente usada no DML (com logs e dummies) |
| `dml_resultados.csv` | θ̂, SE, IC95%, *z* e *p*-valor para LASSO e RF |
| `dml_residuos.csv` | Resíduos *out-of-fold* das equações auxiliares |
| `dml_forest_plot.png` | Forest plot das estimativas |
| `importancia_variaveis_rf.csv` | Importância das variáveis em ĝ e m̂ (RF) |

---

## Resultados esperados

Os valores numéricos exatos dependem de versões dos pacotes, do
*seed*, e da partição K-fold, mas os padrões substantivos esperados
são:

- **θ̂ positivo** nos dois aprendizes (LASSO e Random Forest),
  indicando que municípios em anos com maior valor contratado de
  PRONAF tendem a apresentar, em média, **maior produtividade do
  milho**, controlando por clima, solo, estrutura econômica, região e
  ano. O milho é cultura amplamente difundida no Brasil, com forte
  presença da agricultura familiar (cesta tradicional do PRONAF), o
  que faz desta uma aplicação substantiva natural do método.
- **Convergência entre LASSO e Random Forest** dentro do IC95%,
  ainda que o Random Forest tipicamente capture mais não-linearidades
  e produza um SE ligeiramente diferente.


> **Atenção interpretativa.** A magnitude de θ̂ deve ser lida como
> *variação esperada em log(produtividade) por unidade de aumento em
> log(1 + PRONAF)*, **sob a hipótese de ignorabilidade condicional em
> X**. Veja também o caveat na seção
> [Modelo e método](#modelo-e-método): este exemplo ilustra a
> mecânica do DML em painel, não substitui uma estratégia completa de
> identificação causal.

---

## Solução de problemas

### Windows: `Rscript` não é reconhecido (PowerShell)

> *"O termo 'Rscript' não é reconhecido como nome de cmdlet..."*

O R está instalado mas não está no PATH. Use o caminho completo:

```powershell
& "C:\Program Files\R\R-4.4.2\bin\Rscript.exe" R\exemplo3.R dados\base_pronaf.csv
```

Ou, para adicionar ao PATH permanentemente:

```powershell
[Environment]::SetEnvironmentVariable(
  "Path",
  [Environment]::GetEnvironmentVariable("Path","User") + ";C:\Program Files\R\R-4.4.2\bin",
  "User"
)
```

Como alternativa, use a [Opção B](#opção-b--dentro-do-r--rstudio-recomendada-no-windows).

### macOS / Linux: `Rscript: command not found`

```bash
# Debian/Ubuntu
sudo apt-get install r-base

# Fedora
sudo dnf install R

# macOS (Homebrew)
brew install r
```

### O Random Forest demora muito ou estoura memória

Reduza `num.trees = 500` para 200 ou 100 nas chamadas a `regr.ranger`,
ou reduza `n_folds = 5` para 3 em `DoubleMLPLR$new(...)`. A comparação
relativa entre LASSO e RF se mantém.

### Pacotes do mlr3 não instalam (problema de dependência)

A família mlr3 tem muitas dependências encadeadas. Instale-as todas
de uma vez:

```r
install.packages(c("mlr3", "mlr3learners", "mlr3pipelines",
                   "mlr3tuning", "paradox", "checkmate", "lgr"))
```

### Aviso *"prediction had a NA value"* nas equações auxiliares

Geralmente decorre de uma covariável com poucas observações em algum
fold do cross-fitting. Tente reduzir o número de folds
(`n_folds = 3`) ou remover variáveis com excesso de zeros.

---

## Como citar

Se você utilizar este material em pesquisa ou ensino, por favor cite o
capítulo do livro:

>
> (no prelo)
>

e a referência fundadora do método:

> Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen,
> C., Newey, W., & Robins, J. (2018). Double/debiased machine learning
> for treatment and structural parameters. *The Econometrics Journal*,
> 21(1), C1–C68.

E mencione este repositório:

> Repositório de apoio ao Exemplo 3 — Efeito do PRONAF sobre a
> produtividade do milho municipal via Double/Debiased Machine
> Learning, painel 2013-2021. Disponível em:
> `https://github.com/luizsatolo/livro-les-cap-ml (pasta `exemplo3-pronaf-dml/`)`

---

## Licença

- **Código**: MIT (ver arquivo [`LICENSE`](LICENSE)).
- **Dados**: Creative Commons Attribution 4.0 International (CC-BY-4.0),
  conforme as licenças das fontes (IBGE/PAM, ERA5/Copernicus,
  MapBiomas, SICOR/Bacen).

---

## Contato

Dúvidas, sugestões ou problemas de reprodução: abra uma *issue* neste
repositório.
