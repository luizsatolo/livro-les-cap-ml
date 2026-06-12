# Exemplo 2 — Classificação de (in)segurança alimentar em domicílios brasileiros (POF 2017-2018)

Material de apoio do **Exemplo 2** do capítulo *"Machine Learning em
Avaliação de Impacto na Economia"*. O exemplo compara dois modelos de
classificação multinomial — **regressão logística multinomial** e
**Random Forest** — para prever a classe da **Escala Brasileira de
Insegurança Alimentar (EBIA)** em domicílios da Pesquisa de Orçamentos
Familiares (POF) 2017-2018 do IBGE.

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

A insegurança alimentar é um problema social persistente no Brasil. A
**EBIA** é o instrumento oficial brasileiro para medir, em escala
domiciliar, a percepção dos moradores sobre o acesso aos alimentos nos
últimos três meses. Ela classifica os domicílios em quatro classes em
ordem crescente de gravidade: *segurança alimentar*, *insegurança
alimentar leve*, *moderada* e *grave*.

Aplicar a EBIA exige um módulo específico de 14 perguntas, o que torna
sua coleta cara e pouco frequente. Modelos de aprendizado de máquina
treinados em pesquisas que aplicaram a EBIA (como a POF 2017-2018) podem
ser úteis para *imputar* a classe da EBIA em pesquisas que não a
coletaram, mas registram características demográficas e socioeconômicas
do domicílio.

Este exemplo mostra como treinar e avaliar dois classificadores
multinomiais com esse propósito, dando atenção tanto à **acurácia
preditiva** quanto à **interpretabilidade dos coeficientes** (via odds
ratios da regressão logística e importância de variáveis do Random
Forest).

---

## Modelos comparados

Os dois modelos são treinados nos mesmos 80% da amostra (split
estratificado por `classe_ebia`) e avaliados nos 20% restantes.

| # | Modelo | Especificação | O que captura |
|---|---|---|---|
| 1 | **Regressão logística multinomial** | `multinom(classe_ebia ~ ., ...)` (`nnet::multinom`) | Relações log-lineares entre covariáveis e classe; produz odds ratios diretamente interpretáveis. |
| 2 | **Random Forest** | `ranger(classe_ebia ~ ., num.trees = 500, mtry = floor(sqrt(p)), ...)` (`ranger::ranger`) | Não-linearidades e interações entre variáveis; produz escores de importância (impurity). |

**Métricas de avaliação** (no conjunto de teste): *accuracy*, *F1 macro*,
*balanced accuracy macro* e *Kappa de Cohen*.

---

## Estrutura do repositório

Esta pasta é parte do repositório consolidado [`livro-les-cap-ml`](https://github.com/luizsatolo/livro-les-cap-ml) e segue a estrutura abaixo dentro de `exemplo2-ebia-pof/`:

```
exemplo2-ebia-pof/
├── R/
│   └── exemplo2.R                # script principal (Exemplo 2)
├── dados/
│   └── base_ebia.csv             # POF 2017-2018, base domiciliar
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
| `janitor` | padronização dos nomes das colunas (`clean_names`) |
| `forcats` | manipulação de fatores |
| `caret` | partição estratificada treino/teste (`createDataPartition`) |
| `nnet` | regressão logística multinomial (`multinom`) |
| `ranger` | Random Forest multiclasses |
| `yardstick` | métricas (accuracy, F1, Kappa, etc.) |
| `ggplot2` | gráficos |
| `tibble` | tibbles |

Para instalar manualmente, antes de executar:

```r
install.packages(c(
  "readr", "dplyr", "janitor", "forcats", "caret",
  "nnet", "ranger", "yardstick", "ggplot2", "tibble"
))
```

---

## Como reproduzir o exemplo

### Opção A — Linha de comando

```bash
git clone https://github.com/luizsatolo/livro-les-cap-ml.git
cd livro-les-cap-ml/exemplo2-ebia-pof.git
Rscript R/exemplo2.R dados/base_ebia.csv
```

### Opção B — Dentro do R / RStudio (recomendada no Windows)

```r
# defina o diretório de trabalho na raiz do repositório
setwd("caminho/para/o/repositorio")

# executa o script
source("R/exemplo2.R")
```

> Se executado sem argumento, o script tenta ler `dados/base_ebia.csv`
> a partir do diretório de trabalho atual.

A execução completa leva tipicamente entre 1 e 5 minutos, dependendo da
máquina (o Random Forest com 500 árvores é o passo mais demorado).

---

## A base de dados

O arquivo `dados/base_ebia.csv` contém **57.920 domicílios** da Pesquisa
de Orçamentos Familiares (POF) 2017-2018 do IBGE, com 16 variáveis:

- **Variável dependente** — `classe_ebia`, derivada do módulo EBIA
  aplicado na POF. As quatro classes seguem a definição oficial da
  Escala Brasileira de Insegurança Alimentar.
- **Identificadores e plano amostral** — `uf`, `estrato_pof`, `cod_upa`
  e `num_dom`. Não entram nos modelos.
- **Características do domicílio** — situação (urbano/rural), tipo de
  edificação, abastecimento de água, esgoto, destino do lixo, energia
  elétrica, número de cômodos, banheiros e moradores.
- **Renda e fontes de rendimento** — renda per capita (em R$) e número
  de fontes de rendimento. No script, a renda per capita é convertida
  para R$ mil para facilitar a interpretação dos odds ratios.

> **Fonte:** IBGE — POF 2017-2018, microdados abertos
> (https://www.ibge.gov.br/estatisticas/sociais/saude/24786-pesquisa-de-orcamentos-familiares-2.html).
> O arquivo CSV deste repositório foi pré-processado a partir dos
> microdados originais para conter, por linha, um domicílio com sua
> classificação da EBIA e suas características demográficas e
> socioeconômicas.

---

## Dicionário de variáveis

| Variável | Tipo | Papel | Descrição | Unidade/níveis |
|---|---|---|---|---|
| `uf` | inteiro | identificador (excluído do modelo) | Código da Unidade da Federação | 11–53 |
| `estrato_pof` | inteiro | plano amostral (excluído) | Estrato de amostragem da POF | — |
| `cod_upa` | inteiro | plano amostral (excluído) | Código da Unidade Primária de Amostragem | — |
| `num_dom` | inteiro | identificador (excluído) | Número sequencial do domicílio na UPA | — |
| `classe_ebia` | fator (4 níveis) | **dependente** | Classificação EBIA do domicílio | Segurança / Insegurança leve / moderada / grave |
| `situacao_dom` | fator | preditor | Situação do domicílio | Urbano / Rural |
| `tipo_dom` | fator | preditor | Tipo de domicílio | Casa / Apartamento / outros |
| `abastecimento_agua` | fator | preditor | Forma de abastecimento de água | Rede geral / Poço / outros |
| `esgoto_sanitario` | fator | preditor | Tipo de esgotamento sanitário | Rede / Fossa / outros |
| `destino_lixo` | fator | preditor | Destino do lixo | Coleta / Outros |
| `energia_eletrica` | fator | preditor | Existência de energia elétrica | Sim / Não |
| `num_comodos` | inteiro | preditor | Número de cômodos do domicílio | — |
| `num_banheiros` | inteiro | preditor | Número de banheiros | — |
| `num_moradores` | inteiro | preditor | Número de moradores | — |
| `fontes_renda` | inteiro | preditor | Número de fontes de rendimento | — |
| `renda_percapita` | numérica | preditor (convertida) | Renda per capita | R$ (convertida a R$ mil no script: `percapita_mil`) |

---

## O que o script faz, passo a passo

O script `R/exemplo2.R` está organizado em 16 seções comentadas. Em
resumo:

1. **Carrega pacotes** e fixa a semente aleatória (`set.seed(123)`).
2. **Lê argumentos** da linha de comando e cria o diretório de saída
   (`resultados_modelos/`).
3. **Lê e valida** a base; aborta se faltar `classe_ebia`.
4. **Prepara os dados**: define `classe_ebia` como fator com ordem
   crescente de gravidade; exclui identificadores e variáveis do plano
   amostral (`uf`, `estrato_pof`, `cod_upa`, `num_dom`); converte
   `renda_percapita` para R$ mil (`percapita_mil`); remove colunas
   totalmente NA e constantes.
5. **Trata valores ausentes**: imputa mediana em variáveis numéricas e
   atribui o nível `"Ignorado"` aos NAs em fatores (exceto na variável
   dependente).
6. **Distribuição das classes**: imprime e salva `distribuicao_classes.csv`.
7. **Split treino/teste**: 80/20 estratificado por `classe_ebia` via
   `caret::createDataPartition` (precedido de `set.seed(123)`); alinha
   níveis de fatores entre treino e teste.
8. **Funções auxiliares**: `avaliar_modelo()` produz uma linha com
   accuracy, F1 macro, balanced accuracy macro e Kappa.
9. **Regressão logística multinomial** (`nnet::multinom`) treinada no
   treino; gera previsões no teste.
10. **Random Forest** (`ranger::ranger` com 500 árvores,
    `mtry = floor(sqrt(p))`, `min.node.size = 10`, `seed = 123`); gera
    previsões no teste.
11. **Métricas comparativas** salvas em `metricas_classificacao.csv`.
12. **Matrizes de confusão** dos dois modelos (csv + impressão em
    formato `yardstick::conf_mat`).
13. **Importância de variáveis (RF)**: escores impurity, csv e gráfico
    das 15 variáveis mais importantes.
14. **Odds ratios da logística** para as cinco variáveis mais
    importantes do RF: extração de coeficientes e erros-padrão,
    IC95% pelo método de Wald, exportação em csv e *forest plot* em
    PNG.
15. **Previsões no teste** com classe observada e previsões dos dois
    modelos.
16. Mensagem final indicando onde os resultados foram salvos.

---

## Saídas geradas

Todas as saídas são gravadas em `resultados_modelos/`:

| Arquivo | Conteúdo |
|---|---|
| `distribuicao_classes.csv` | Frequência e proporção de cada classe da EBIA |
| `metricas_classificacao.csv` | Accuracy, F1 macro, Bal. accuracy e Kappa por modelo |
| `matriz_confusao_logit.csv` | Matriz de confusão (verdade × previsto) da logística |
| `matriz_confusao_rf.csv` | Matriz de confusão do Random Forest |
| `importancia_variaveis_rf.csv` | Escore impurity por variável (decrescente) |
| `importancia_variaveis_rf.png` | Gráfico das 15 variáveis mais importantes |
| `odds_ratios_logit.csv` | OR e IC95% das 5 variáveis mais importantes do RF |
| `odds_ratios_logit.png` | Forest plot dos odds ratios |
| `previsoes_teste.csv` | Classe observada e previsões dos dois modelos no teste |

---

## Resultados esperados

Os valores exatos podem variar ligeiramente em função da versão dos
pacotes e da máquina, mas a **ordem de mérito** entre os modelos tende
a se manter:

- A **regressão logística multinomial** mantém desempenho competitivo
  na classe majoritária ("Segurança alimentar") e oferece a vantagem
  de produzir odds ratios diretamente interpretáveis.
- O **Random Forest** apresenta macro F1, acurácia balanceada e Kappa 
  mais altos do que a regressão logística, refletindo sua capacidade de 
  capturar não-linearidades e interações.
- Ambos os modelos têm dificuldade maior nas classes minoritárias
  ("moderada" e "grave"), o que é típico em amostras com forte
   desbalanceamento.

A discussão completa dos resultados está no capítulo do livro.

---

## Solução de problemas

### Windows: `Rscript` não é reconhecido (PowerShell ou Prompt de Comando)

Mensagem típica no PowerShell:

> *"O termo 'Rscript' não é reconhecido como nome de cmdlet, função,
> arquivo de script ou programa operável."*

Esse erro significa que o R **está instalado** (ou ainda não está) e que
a pasta `bin` do R não foi adicionada ao `PATH` do sistema. Três saídas:

**(a) Diagnóstico — descobrir se o R está instalado e onde.** No
PowerShell:

```powershell
Get-ChildItem "C:\Program Files\R\R-*\bin\Rscript.exe" -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty FullName
```

- Se aparecer um caminho, o R está instalado — siga para (b) ou (c).
- Se não aparecer nada, instale o R em
  https://cran.r-project.org/bin/windows/base/ .

**(b) Solução imediata — usar o caminho completo.** Na pasta do
repositório:

```powershell
& "C:\Program Files\R\R-4.4.2\bin\Rscript.exe" R\exemplo2.R dados\base_ebia.csv
```

**(c) Solução definitiva — adicionar o R ao `PATH`.** Permanente para
o usuário (rodar uma vez; depois reabrir o PowerShell):

```powershell
[Environment]::SetEnvironmentVariable(
  "Path",
  [Environment]::GetEnvironmentVariable("Path","User") + ";C:\Program Files\R\R-4.4.2\bin",
  "User"
)
```

**(d) Alternativa sem linha de comando.** Use a
[Opção B](#opção-b--dentro-do-r--rstudio-recomendada-no-windows). O
RStudio chama o R com o caminho correto internamente.

### macOS / Linux: `Rscript: command not found`

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
  "readr", "dplyr", "janitor", "forcats", "caret",
  "nnet", "ranger", "yardstick", "ggplot2", "tibble"
))
```

### O Random Forest demora muito ou estoura memória

Reduza o número de árvores em `ranger(num.trees = 500, ...)` para 200 ou
100. O resultado piora ligeiramente, mas a comparação relativa entre os
modelos se mantém.

### Aviso *"glm.fit: fitted probabilities numerically 0 or 1 occurred"*

Pode ocorrer na regressão logística quando uma variável é quase
perfeitamente separável (ex.: `energia_eletrica = "Não"` é raro e
correlacionado com insegurança grave). É apenas um aviso e não
compromete os resultados; se incomodar, agrupe categorias raras antes
de estimar o modelo.

---

## Como citar

Se você utilizar este material em pesquisa ou ensino, por favor cite o
capítulo do livro:

>
> (no prelo)
>

e mencione este repositório:

> Repositório de apoio ao Exemplo 2 — Classificação de (in)segurança
> alimentar em domicílios brasileiros (POF 2017-2018). Disponível em:
> `https://github.com/luizsatolo/livro-les-cap-ml.git (pasta `exemplo2-ebia-pof/`)`

---

## Licença

- **Código**: MIT (ver arquivo [`LICENSE`](LICENSE)).
- **Dados**: Creative Commons Attribution 4.0 International (CC-BY-4.0),
  conforme a política de microdados do IBGE.

---

## Contato

Dúvidas, sugestões ou problemas de reprodução: abra uma *issue* neste
repositório.
