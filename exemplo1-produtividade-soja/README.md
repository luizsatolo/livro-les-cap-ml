# Exemplo 1 — Previsão da produtividade da soja em municípios brasileiros

Material de apoio do **Exemplo 1** do capítulo *"Machine Learning em Avaliação
de Impacto na Economia"*. O exemplo compara três estratégias para prever a
produtividade da soja (kg/ha) em municípios brasileiros no período
2004–2021, combinando técnicas de econometria de painel com regressão
penalizada.

---

## Sumário

1. [Contexto](#contexto)
2. [Modelos comparados](#modelos-comparados)
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

Modelos de aprendizado de máquina vêm sendo cada vez mais utilizados em
avaliação de impacto na economia, particularmente em problemas em que existem
muitas covariáveis potencialmente relevantes (alta dimensionalidade) e
estruturas de painel longitudinais. A produtividade agrícola municipal é um
exemplo clássico: depende simultaneamente de fatores climáticos,
edáficos, demográficos e econômicos, com forte heterogeneidade entre
municípios e dinâmica temporal.

O objetivo deste exemplo é mostrar, de forma reprodutível, como combinar
**efeitos fixos** com **regressão penalizada (LASSO)** para obter previsões
mais precisas do que cada um dos dois métodos isoladamente. O exercício é
construído sobre dados públicos abertos.

---

## Modelos comparados

Os três modelos são estimados na **mesma amostra comum**, garantindo
comparabilidade de métricas.

| # | Modelo | Especificação | O que captura |
|---|---|---|---|
| 1 | **LASSO puro** | `produtividade ~ X` com penalização L1 (`cv.glmnet`, `lambda.1se`) | Estrutura global entre covariáveis e produtividade, com seleção automática de variáveis. |
| 2 | **FE Dinâmico** | `produtividade ~ produtividade_lag \| id_municipio` (`fixest::feols`) | Heterogeneidade municipal não observada (efeitos fixos) + inércia temporal (lag de 1 ano). |
| 3 | **Híbrido FE + LASSO** | FE Dinâmico nos níveis + LASSO sobre o resíduo do FE | Combina a estrutura de painel com a capacidade do LASSO de capturar relações não modeladas pelo FE. |

**Estratégia de avaliação:** *split temporal* — treino com anos
2004–2018 e teste com 2019–2021. As métricas reportadas (RMSE, MAE e R²)
são calculadas no conjunto de teste.

---

## Estrutura do repositório

Esta pasta é parte do repositório consolidado [`livro-les-cap-ml`](https://github.com/luizsatolo/livro-les-cap-ml) e segue a estrutura abaixo dentro de `exemplo1-produtividade-soja/`:

```
exemplo1-produtividade-soja/
├── R/
│   └── exemplo1.R                # script principal (Exemplo 1)
├── dados/
│   └── base_final.csv            # painel municipal 2004–2024
├── resultados_modelos/           # criada/atualizada na execução
└── README.md
```

---

## Requisitos

- **R** ≥ 4.1 (recomendado 4.3 ou superior; o script usa o pipe nativo `|>`).
- **RStudio** (opcional, mas recomendado para inspeção interativa).
- Conexão à internet na primeira execução (instala pacotes ausentes do CRAN).

### Pacotes R

O script instala automaticamente qualquer pacote ausente. A lista completa é:

| Pacote | Função no exemplo |
|---|---|
| `readr` | leitura do CSV |
| `dplyr` | manipulação de data frames |
| `tidyr` | pivots e remoção de NAs |
| `glmnet` | LASSO via validação cruzada (`cv.glmnet`) |
| `fixest` | efeitos fixos via `feols` |
| `yardstick` | métricas RMSE, MAE e R² |
| `tibble` | tibbles |
| `ggplot2` | gráficos de avaliação |

Para instalar manualmente, antes de executar:

```r
install.packages(c(
  "readr", "dplyr", "tidyr", "glmnet",
  "fixest", "yardstick", "tibble", "ggplot2"
))
```

---

## Como reproduzir o exemplo

### Opção A — Linha de comando (recomendada)

```bash
git clone https://github.com/luizsatolo/livro-les-cap-ml.git
cd livro-les-cap-ml/exemplo1-produtividade-soja
Rscript R/exemplo1.R dados/base_final.csv
```

### Opção B — Dentro do R / RStudio

```r
# defina o diretório de trabalho na raiz do repositório
setwd("caminho/para/o/repositorio")

# executa passando o caminho da base como argumento
source("R/exemplo1.R")
```

> Se executado sem argumento, o script tenta ler `dados/base_final.csv` a
> partir do diretório de trabalho atual.

A execução completa leva tipicamente entre 1 e 5 minutos, dependendo da
máquina.

---

## A base de dados

O arquivo `dados/base_final.csv` é um painel municipal com cerca de **44.500
observações** referentes a aproximadamente **2.800 municípios** brasileiros
entre 2004 e 2024. O exemplo utiliza apenas o recorte 2004–2021.

As variáveis foram organizadas a partir de **dados públicos abertos**:

- **Produtividade da soja** — IBGE, Produção Agrícola Municipal (PAM).
- **Variáveis climáticas** — base [**ERA5**](https://cds.climate.copernicus.eu/)
  (reanálise atmosférica do ECMWF/Copernicus). Foram extraídas as
  séries diárias por município (agregação por área) e calculados, para
  cada ano, os indicadores anuais utilizados no exemplo: temperatura
  média, temperatura máxima, dias acima do percentil 95 de temperatura
  média (TX95p), precipitação acumulada anual, precipitação máxima em 5
  dias consecutivos (Rx5day) e máxima sequência de dias consecutivos
  secos (CDD).
- **Variáveis edáficas** — projeto [**MapBiomas**](https://brasil.mapbiomas.org/)
  (carbono médio do solo e teores de argila e areia na camada
  0–10 cm).
- **Variáveis socioeconômicas** — IBGE (PIB total e setorial, PIB per
  capita, população, área e densidade demográfica).

> **Atenção:** o arquivo CSV foi pré-processado a partir das fontes acima
> para facilitar a reprodução. Para os scripts originais de extração e
> harmonização, consulte o repositório do autor (em construção).

---

## Dicionário de variáveis

| Variável | Tipo | Fonte | Descrição | Unidade |
|---|---|---|---|---|
| `id_municipio` | identificador | IBGE | Código do município (7 dígitos) | — |
| `ano` | inteiro | — | Ano de referência | — |
| `produtividade_soja` | numérica | IBGE/PAM | **Variável-alvo:** produtividade da soja | kg/ha |
| `temperatura_media` | numérica | ERA5 | Temperatura média anual (média municipal) | °C |
| `temperatura_maxima` | numérica | ERA5 | Temperatura máxima anual (média municipal) | °C |
| `dias_quentes` | numérica | ERA5 | Dias com temperatura média acima do percentil 95 da climatologia (índice TX95p) | dias/ano |
| `precipitacao_anual` | numérica | ERA5 | Precipitação acumulada anual | mm |
| `precipitacao_extrema` | numérica | ERA5 | Máxima precipitação acumulada em 5 dias consecutivos (índice Rx5day) | mm |
| `dias_secos` | numérica | ERA5 | Maior sequência de dias consecutivos secos no ano (índice CDD) | dias |
| `carbono_solo` | numérica | MapBiomas | Carbono médio no solo | g/kg |
| `teor_argila` | numérica | MapBiomas | Teor de argila (camada 0–10 cm) | % |
| `teor_areia` | numérica | MapBiomas | Teor de areia (camada 0–10 cm) | % |
| `pib_total` | numérica | IBGE | PIB municipal total | R$ mil |
| `pib_agropecuaria` | numérica | IBGE | PIB do setor agropecuário | R$ mil |
| `pib_industria` | numérica | IBGE | PIB do setor industrial | R$ mil |
| `pib_servicos` | numérica | IBGE | PIB do setor de serviços | R$ mil |
| `populacao_total` | numérica | IBGE | População total do município | habitantes |
| `area_municipio` | numérica | IBGE | Área do município | km² |
| `pib_percapita` | numérica | IBGE | PIB per capita | R$ mil/hab |
| `densidade_demografica` | numérica | IBGE | Habitantes por km² | hab/km² |
| `produtividade_lag` | numérica (criada no script) | — | Produtividade da soja defasada em 1 ano | kg/ha |

---

## O que o script faz, passo a passo

O script `R/exemplo1.R` está organizado em 17 seções comentadas. Em resumo:

1. **Carrega pacotes** e fixa a semente aleatória (`set.seed(123)`).
2. **Lê argumentos** da linha de comando e define o diretório de saída
   (`resultados_modelos/`).
3. **Lê e valida** a base; aborta se faltarem `id_municipio`, `ano` ou
   `produtividade_soja`.
4. **Prepara os dados**: converte tipos, filtra 2004–2021, remove colunas
   textuais não identificadoras e cria `produtividade_lag` por município.
5. **Split temporal**: treino ≤ 2018; teste 2019–2021 (mantendo apenas
   municípios presentes no treino).
6. **Seleciona covariáveis numéricas** e elimina as constantes no treino.
7. **Imputa medianas** (calculadas no treino) para covariáveis numéricas,
   exceto o lag.
8. Define **funções auxiliares** para métricas, extração de coeficientes
   não nulos e construção da matriz de design.
9. **Define a amostra comum** de forma puramente determinística (drop_na,
   remoção de municípios singleton e construção da matriz de design `X` a
   partir de `produtividade ~ vars`). Essa amostra é a mesma para os três
   modelos, garantindo que a ordem de estimação não interfira nos
   resultados.
10. **LASSO puro** na amostra comum (`cv.glmnet` precedido de
    `set.seed(123)`).
11. **FE Dinâmico** na amostra comum
    (`feols(produtividade ~ produtividade_lag | id_municipio)`). Uma
    assertiva verifica que o FE não introduz `NA` em nenhuma linha da
    amostra comum.
12. **Híbrido FE + LASSO**: usa o FE do passo anterior para calcular o
    resíduo no treino e aplica `cv.glmnet` (também precedido de
    `set.seed(123)`) sobre esse resíduo, reaproveitando a mesma matriz `X`.
    A previsão final é o ajuste do FE no teste somado à previsão LASSO do
    resíduo.
13. **Métricas** (RMSE, MAE, R²) no conjunto de teste para os três modelos.
14. **Salva** métricas, previsões e coeficientes não nulos do LASSO e do
    LASSO do híbrido.
15. **Gráfico 1**: dispersão observado vs. previsto, com faceta por modelo.
16. **Gráfico 2**: série temporal observado e previsto nos municípios de
    menor e maior produtividade média no teste e na média municipal.
17. Mensagem final indicando onde os resultados foram salvos.

---

## Saídas geradas

Todas as saídas são gravadas em `resultados_modelos/`:

| Arquivo | Conteúdo |
|---|---|
| `metricas_comparacao.csv` | RMSE, MAE e R² por modelo no teste |
| `previsoes_comparacao.csv` | Predições municipais no teste para os três modelos |
| `coeficientes_lasso.csv` | Coeficientes não nulos do LASSO puro (em `lambda.1se`) |
| `coeficientes_hibrido.csv` | Coeficientes não nulos do LASSO do modelo híbrido |
| `observado_vs_previsto.png` | Gráfico de dispersão observado × previsto |
| `serie_municipios_extremos.png` | Série temporal nos extremos e na média |

---

## Resultados esperados

Os números exatos podem variar ligeiramente em função da versão dos pacotes
e do número de *folds* da validação cruzada, mas a **ordem de mérito** entre
os modelos tende a se manter:

- O **LASSO puro** apresenta bom desempenho em termos de erro de previsão.
- O **FE Dinâmico** captura bem a heterogeneidade municipal.
- O **Híbrido FE + LASSO** entrega o menor RMSE/MAE e o maior R² no teste.


A interpretação detalhada dos resultados está no capítulo do livro.

---

## Solução de problemas

### Windows: `Rscript` não é reconhecido (PowerShell ou Prompt de Comando)

Mensagem típica no PowerShell:

> *"O termo 'Rscript' não é reconhecido como nome de cmdlet, função,
> arquivo de script ou programa operável."*

Esse erro significa que o R **está instalado** (ou ainda não está) e que a
pasta `bin` do R não foi adicionada ao `PATH` do sistema. Existem três
saídas:

**(a) Diagnóstico — descobrir se o R está instalado e onde.** Cole no
PowerShell:

```powershell
Get-ChildItem "C:\Program Files\R\R-*\bin\Rscript.exe" -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty FullName
```

- Se aparecer um caminho (por exemplo `C:\Program Files\R\R-4.4.2\bin\Rscript.exe`),
  o R está instalado — siga para (b) ou (c).
- Se não aparecer nada, instale o R a partir de
  https://cran.r-project.org/bin/windows/base/ (basta aceitar as opções
  padrão).

**(b) Solução imediata — usar o caminho completo do `Rscript.exe`.** Dentro
da pasta do repositório, no PowerShell:

```powershell
& "C:\Program Files\R\R-4.4.2\bin\Rscript.exe" R\exemplo1.R dados\base_final.csv
```

O `&` é o operador de chamada do PowerShell, necessário porque o caminho
do executável contém espaços. Ajuste a versão do R conforme o resultado
do diagnóstico.

**(c) Solução definitiva — adicionar o R ao `PATH`.**

Para a sessão atual do PowerShell:

```powershell
$env:Path += ";C:\Program Files\R\R-4.4.2\bin"
Rscript R\exemplo1.R dados\base_final.csv
```

Permanente para o usuário (executar uma vez; depois reabrir o PowerShell):

```powershell
[Environment]::SetEnvironmentVariable(
  "Path",
  [Environment]::GetEnvironmentVariable("Path","User") + ";C:\Program Files\R\R-4.4.2\bin",
  "User"
)
```

**(d) Alternativa sem linha de comando.** Abra o RStudio e use a
[Opção B](#opção-b--dentro-do-r--rstudio) descrita acima. O RStudio chama
o R com o caminho correto internamente.

### macOS / Linux: `Rscript: command not found`

Instale o R via gestor de pacotes da distribuição:

```bash
# Debian/Ubuntu
sudo apt-get install r-base

# Fedora
sudo dnf install R

# macOS (Homebrew)
brew install r
```

### Pacotes não instalam pela primeira execução (sem internet ou proxy)

Instale-os manualmente uma única vez dentro do R:

```r
install.packages(c(
  "readr", "dplyr", "tidyr", "glmnet",
  "fixest", "yardstick", "tibble", "ggplot2"
))
```

Depois rode o script normalmente — ele detecta que os pacotes já estão
disponíveis e pula a instalação.

### `Error: cannot allocate vector of size ...`

Indica falta de memória RAM. A base tem ~44 mil linhas e o script foi
testado em máquinas com 8 GB de RAM, mas se ocorrer:

- Feche outros programas pesados;
- Reduza o número de *folds* alterando `nfolds = 10` para `nfolds = 5`
  nas duas chamadas de `cv.glmnet` no script.

### O script aborta com *"FE retornou NAs em algum município da amostra comum"*

Significa que a base foi editada e algum município ficou sem o número
mínimo de observações no treino exigido pelo FE. Refaça o filtro inicial
(`ano >= 2004 & ano <= 2021`) e verifique se todos os municípios do teste
estão presentes no treino. Os passos da seção 9 do script reproduzem
exatamente essa lógica.

---

## Como citar

Se você utilizar este material em pesquisa ou ensino, por favor cite o
capítulo do livro:

>
> (no prelo)
>

e mencione este repositório:

> Repositório de apoio ao Exemplo 1 — Previsão da produtividade da soja
> em municípios brasileiros. Disponível em:
> `https://github.com/luizsatolo/livro-les-cap-ml.git (pasta `exemplo1-produtividade-soja/`)`

---

## Licença

- **Código**: MIT (ver arquivo [`LICENSE`](LICENSE)).
- **Dados**: Creative Commons Attribution 4.0 International (CC-BY-4.0),
  conforme as licenças originais das fontes públicas que deram origem ao
  painel (IBGE, ERA5/Copernicus e MapBiomas).

---

## Contato

Dúvidas, sugestões ou problemas de reprodução: abra uma *issue* neste
repositório.
