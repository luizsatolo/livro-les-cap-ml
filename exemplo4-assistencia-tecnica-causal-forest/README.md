# Exemplo 4 — Efeitos heterogêneos da assistência técnica agrícola sobre a produtividade agropecuária municipal via Causal Forest (cross-section 2017)

Material de apoio do **Exemplo 4** do capítulo *"Machine Learning em
Avaliação de Impacto na Economia"*, ilustrando a seção 5.3
(Causal Forest e Efeitos Heterogêneos do Tratamento). O exemplo estima,
em **cross-section municipal de 2017**, o efeito da exposição à
**assistência/orientação técnica** sobre a **produtividade
agropecuária** dos municípios brasileiros, com foco no **CATE**
(*Conditional Average Treatment Effect*) — o efeito do tratamento
condicional aos atributos municipais — via algoritmo *Causal Forest*
de Wager e Athey (2018).

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

Estimadores convencionais de avaliação de impacto identificam efeitos
médios — o ATE (*Average Treatment Effect*) e o ATT (*Average Treatment
Effect on the Treated*). Em muitas aplicações, porém, o efeito médio é
insuficiente: políticas que produzem efeitos positivos em um subgrupo
podem ser irrelevantes ou até contraproducentes em outro. Assistência
técnica agrícola é um caso clássico — sua eficácia tende a depender de
tamanho da propriedade, clima, perfil produtivo e acesso a outros
insumos.

A abordagem convencional para investigar essa heterogeneidade é incluir
interações entre tratamento e subgrupos pré-definidos na regressão.
Essa estratégia tem dois problemas: o pesquisador precisa **escolher a
priori** os subgrupos relevantes (risco de busca oportunista) e o
**poder estatístico** cai quando o número de subgrupos é grande.

A **Causal Forest** (Wager e Athey, 2018; Athey e Wager, 2019) resolve
o problema. É uma adaptação do *Random Forest* em que cada árvore é
construída para **maximizar a heterogeneidade dos efeitos do tratamento
entre as regiões resultantes**, e não para maximizar a homogeneidade da
variável de resposta. Dentro de cada folha, o efeito é estimado por
diferença de médias entre tratados e controles. As árvores são
**honestas**: usam subamostras distintas para escolher os splits e
para estimar τ(x) dentro de cada folha, o que confere validade
assintótica à inferência.

Aqui aplicamos Causal Forest em um *cross-section municipal de 2017*
para estimar o efeito heterogêneo da exposição à orientação técnica
sobre a produtividade agropecuária, condicional a clima (ERA5), solo
(MapBiomas), estrutura agrícola (Censo Agro 2017), estrutura econômica
(IBGE), exposição ao crédito (PRONAF/SICOR) e macrorregião.

---

## Modelo e método

A Causal Forest implementa o **modelo de efeitos heterogêneos**:

```
Y_i = m(X_i) + W_i · τ(X_i) + ε_i ,   E[ε_i | X_i, W_i] = 0
```

em que:

- `Y_i` = `log(produtividade_agropecuaria)` do município *i* (kg/ha em
  valor monetário/ha, do Censo Agropecuário 2017);
- `W_i` ∈ {0, 1} = 1 se a **proporção de estabelecimentos
  agropecuários do município com orientação técnica recebida** está
  acima da **mediana nacional** em 2017 (≈ 21,5%); 0 caso contrário;
- `X_i` = vetor de 25 covariáveis: 6 climáticas (ERA5), 3 edáficas
  (MapBiomas), 3 de estrutura agrícola (área total, n. de
  estabelecimentos, área média), 7 socioeconômicas (PIB setorial,
  população, densidade, PIB per capita), 2 de crédito
  (`log(1+valor_pronaf)`, número de contratos PRONAF) e 4 *dummies*
  de macrorregião (Norte como referência);
- `τ(X_i)` = **CATE** — efeito causal médio do tratamento condicional
  a `X_i`. É essa função heterogênea que o Causal Forest estima.

### Algoritmo (resumo)

1. A floresta é composta por *B* = 2.000 árvores causais.
2. Cada árvore é construída em uma subamostra aleatória; uma metade
   da subamostra (a **splitting half**) é usada para escolher
   particionamentos do espaço de X que maximizam a diferença estimada
   do efeito do tratamento entre os "filhos" resultantes; a outra
   metade (a **estimating half**) é usada para estimar τ̂ dentro de
   cada folha — esta separação é a propriedade de **honestidade**.
3. Para cada *x* de interesse, τ̂(x) é obtido por uma média
   ponderada dos efeitos folha-a-folha, com pesos derivados de quão
   frequentemente as observações próximas a *x* caem na mesma folha.
4. Sob condições de regularidade, τ̂(x) é assintoticamente normal e
   sua variância é estimável (Wager e Athey, 2018), o que permite
   construção de **intervalos de confiança ponto a ponto**.
5. As funções *nuisance* `m̂(x) = E[Y|X = x]` e `ê(x) = E[W|X = x]`
   são estimadas por *Random Forests* auxiliares com cross-fitting,
   internamente — semelhante ao DML (seção 5.2).

### O que o exemplo reporta

- **ATE *overlap-weighted*, ATT e ATC** — o estimador AIPW padrão
  para `target.sample = "all"` usa peso `1 / [ê(X)·(1−ê(X))]` e
  produz `NaN` quando algum município tem propensão `ê(X)` próxima
  de 0 ou 1; como a exposição à orientação técnica é muito
  assimétrica entre regiões (Sul ~92%, Nordeste ~12%) esse problema
  aparece naturalmente. Reportamos então o **ATE com peso de
  sobreposição** (`ê·(1−ê)`), que dá ênfase à região onde tratados
  e controles coexistem, complementado por **ATT** (peso `1/(1−ê)`)
  e **ATC** (peso `1/ê`), cada um robusto a um dos lados extremos
  da propensão;
- **Teste de calibração de Athey-Wager** — teste formal da hipótese
  nula "não há heterogeneidade" via regressão de Y residualizado em
  τ̂(X) escalado;
- **Importância das variáveis** na partição causal — quais X
  contribuíram mais para os splits que captaram heterogeneidade;
- **Best Linear Projection (BLP)** de τ̂(X) sobre variáveis-chave
  (escala, PIB agropecuário, clima, região), oferecendo coeficientes
  interpretáveis das principais direções de heterogeneidade;
- **CATE por município** com IC95% ponto-a-ponto;
- **CATE médio por subgrupo** — macrorregião e quintis de escala
  (área média do estabelecimento).

> **Caveat de identificação.** A validade do τ̂(X) repousa na
> hipótese de **ignorabilidade condicional em X**:
> `Y(0), Y(1) ⊥ W | X`. Em cross-section observacional, essa
> hipótese é forte — confundidores não capturados em X (qualidade de
> gestão da propriedade, capital social local, herança cultural)
> ainda podem contaminar a estimativa. A leitura adequada do
> exemplo é **descritiva-causal**: que perfis de município tendem a
> apresentar efeitos maiores, condicional ao conjunto observado de
> características.

---

## Estrutura do repositório

Esta pasta é parte do repositório consolidado [`livro-les-cap-ml`](https://github.com/luizsatolo/livro-les-cap-ml) e segue a estrutura abaixo dentro de `exemplo4-assistencia-tecnica-causal-forest/`:

```
exemplo4-assistencia-tecnica-causal-forest/
├── R/
│   └── exemplo4.R                            # script principal (Causal Forest)
├── dados/
│   └── base_assistencia_tecnica.csv          # cross-section 2017 (~5.541 munis)
├── resultados_modelos/                       # criada/atualizada na execução
└── README.md
```

---

## Requisitos

- **R** ≥ 4.1 (recomendado 4.3 ou superior).
- Conexão à internet apenas na primeira execução para instalar
  pacotes ausentes do CRAN. **O script não faz nenhuma chamada de
  rede para acessar dados** — toda a base está em
  `dados/base_assistencia_tecnica.csv`.

### Pacotes R

O script instala automaticamente qualquer pacote ausente:

| Pacote | Função no exemplo |
|---|---|
| `readr` | leitura de CSV |
| `dplyr` | manipulação de data frames |
| `tidyr` | drop_na |
| `tibble` | tibbles |
| `ggplot2` | gráficos |
| `grf` | implementação da Causal Forest (Wager & Athey) |

Para instalar manualmente, antes de executar:

```r
install.packages(c("readr", "dplyr", "tidyr", "tibble", "ggplot2", "grf"))
```

---

## Como reproduzir o exemplo

### Opção A — Linha de comando

```bash
git clone https://github.com/luizsatolo/livro-les-cap-ml.git
cd livro-les-cap-ml/exemplo4-assistencia-tecnica-causal-forest
Rscript R/exemplo4.R dados/base_assistencia_tecnica.csv
```

### Opção B — Dentro do R / RStudio (recomendada no Windows)

```r
# defina o diretório de trabalho na raiz do repositório
setwd("caminho/para/o/repositorio")

# executa o script
source("R/exemplo4.R")
```

A execução típica leva entre 30 segundos e 3 minutos. Por padrão o
script usa 2.000 árvores e `tune.parameters = "all"` (auto-tuning de
profundidade e mtry); reduzir para 500 árvores e desabilitar o
auto-tuning torna a execução praticamente instantânea, à custa de
ligeira perda de precisão.

---

## A base de dados

O arquivo `dados/base_assistencia_tecnica.csv` é um **cross-section
municipal de 2017**, com **5.541 municípios brasileiros** — cobertura
praticamente nacional, exceto pelos poucos municípios sem informação
no Censo Agropecuário ou sem produção agropecuária registrada. As
fontes originais (consolidadas neste único CSV) são:

- **Orientação técnica recebida** (variável de tratamento) — IBGE,
  **Censo Agropecuário 2017** (Tabela SIDRA 6778): número total de
  estabelecimentos agropecuários do município e número dos que
  receberam orientação técnica;
- **Produtividade agropecuária e área dos estabelecimentos** (variável
  de resposta e estrutura agrícola) — IBGE, **Censo Agropecuário
  2017** (valor da produção agropecuária dividido pela área total dos
  estabelecimentos);
- **Variáveis climáticas** — [ERA5](https://cds.climate.copernicus.eu/)
  (reanálise ECMWF/Copernicus), indicadores anuais de 2017;
- **Variáveis edáficas** — [MapBiomas](https://brasil.mapbiomas.org/)
  (carbono médio do solo, teores de argila e areia, camada 0–10 cm);
- **Variáveis socioeconômicas** — IBGE (PIB total e setorial, população,
  área e densidade, PIB per capita), 2017;
- **PRONAF** — SICOR (Sistema de Operações de Crédito Rural) do
  **Banco Central**, valor e número de contratos por município em 2017;
- **Macrorregião** — derivada do código IBGE do município (primeiro
  dígito).

> **Distribuição por macrorregião.** Nordeste 32,2% (1.783 munis,
> 11,9% com alta exposição); Sudeste 29,8% (1.654 munis, 69,7%);
> Sul 21,5% (1.190 munis, 91,5%); Centro-Oeste 8,4% (467 munis, 48,6%);
> Norte 8,1% (447 munis, 19,9%). A forte assimetria regional na
> exposição ao tratamento é, justamente, parte da heterogeneidade
> que o Causal Forest se propõe a capturar.

---

## Dicionário de variáveis

### Identificadores e estrutura

| Variável | Tipo | Descrição |
|---|---|---|
| `id_municipio` | character | Código IBGE do município (7 dígitos) |
| `regiao` | character | Macrorregião (Norte, Nordeste, Sudeste, Sul, Centro-Oeste) |

### Variável de resposta (Y) e tratamento (W)

| Variável | Papel | Fonte | Descrição | Unidade |
|---|---|---|---|---|
| `produtividade_agropecuaria` | base de Y | IBGE/Censo Agro 2017 | Valor da produção agropecuária por hectare dos estabelecimentos | R$/ha |
| `prop_orientacao` | base de W | IBGE/Censo Agro 2017 | Proporção de estabelecimentos com orientação técnica | 0–1 |
| `alta_assistencia` | W | derivado | 1 se `prop_orientacao` > mediana nacional (≈ 0,215) | 0/1 |
| `total_estabelecimentos` | descritor | IBGE/Censo Agro 2017 | Número total de estabelecimentos agropecuários do município | — |
| `estabelecimentos_com_orientacao` | descritor | IBGE/Censo Agro 2017 | Estabelecimentos que receberam orientação técnica | — |

No script:
- `Y = log(produtividade_agropecuaria)`
- `W = alta_assistencia`

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

### Estrutura agrícola (Censo Agropecuário 2017)

| Variável original | Versão usada no Causal Forest |
|---|---|
| `area_estabelecimentos_ha` | `log_area_estab` |
| `total_estabelecimentos` | `log_total_estab` |
| `area_media_estab` (derivada: área total / nº estabelecimentos) | `log_area_media` |

### Estrutura econômica (IBGE)

| Variável original | Versão usada no Causal Forest |
|---|---|
| `pib_total` | `log_pib_total` |
| `pib_agropecuaria` | `log_pib_agropec` |
| `pib_industria` | `log_pib_industria` |
| `pib_servicos` | `log_pib_servicos` |
| `populacao_total` | `log_populacao` |
| `pib_percapita` | `log_pib_percapita` |
| `densidade_demografica` | `densidade_demografica` |

### Crédito (SICOR/Bacen)

| Variável original | Versão usada no Causal Forest |
|---|---|
| `valor_pronaf` | `log_pronaf` (com `log1p` para suportar zeros) |
| `contratos_pronaf` | `contratos_pronaf` |

### Dummies criadas no script

- **Macrorregião** (Norte = referência): `regiao_NE`, `regiao_SE`,
  `regiao_SUL`, `regiao_CO`.

Total de covariáveis em X: 6 (clima) + 3 (solo) + 3 (estrutura
agrícola) + 7 (socio) + 2 (crédito) + 4 (região) = **25**.

---

## O que o script faz, passo a passo

O script `R/exemplo4.R` está organizado em 12 seções comentadas. Em
resumo:

1. **Carrega pacotes** (instala os ausentes) e fixa `set.seed(123)`.
2. **Lê argumentos** da linha de comando e cria o diretório de saída
   (`resultados_modelos/`).
3. **Lê e valida** a base; aborta se faltar alguma variável obrigatória.
4. **Prepara os dados**: filtra `produtividade_agropecuaria > 0`;
   aplica `log` a Y e às variáveis de escala; aplica `log1p` ao
   PRONAF; cria *dummies* de macrorregião.
5. **Especifica Y, W e X** (25 variáveis em X) e salva
   `base_cf_processada.csv` em `resultados_modelos/`.
6. **Ajusta o Causal Forest** com `grf::causal_forest`, 2.000 árvores,
   `honesty = TRUE`, `tune.parameters = "all"`.
7. **Estima ATE, ATT e ATC** (com erros-padrão pelo estimador AIPW
   de `grf::average_treatment_effect`) e roda o **teste de calibração
   de Athey-Wager** para heterogeneidade.
8. **Calcula a importância das variáveis** na partição causal —
   quais X mais explicam a heterogeneidade.
9. **Estima o Best Linear Projection** do CATE sobre um subconjunto
   interpretável de variáveis (escala, PIB agropecuário, clima,
   região).
10. **Prediz o CATE por município** com variância estimada e IC95%;
    agrega o CATE médio por macrorregião e por quintis de escala.
11. **Gera gráficos**: histograma da distribuição dos τ̂ e boxplot do
    CATE por macrorregião.
12. **Mensagem final** indicando onde os resultados foram salvos.

---

## Saídas geradas

Todas as saídas são gravadas em `resultados_modelos/`:

| Arquivo | Conteúdo |
|---|---|
| `base_cf_processada.csv` | Base efetivamente usada no Causal Forest (com logs e dummies) |
| `cf_propensity_overlap.csv` | Quantis do propensity score ê(X) — diagnóstico de overlap |
| `cf_ate.csv` | ATE (overlap-weighted), ATT, ATC com EP, IC95% e *p*-valores |
| `cf_calibration.csv` | Teste de calibração de Athey-Wager |
| `cf_variable_importance.csv` | Importância das variáveis na partição causal |
| `cf_blp.csv` | *Best Linear Projection* do CATE |
| `cf_cate_por_municipio.csv` | CATE estimado, EP e IC95% por município |
| `cf_ate_por_subgrupo.csv` | CATE médio por macrorregião e quintis de escala |
| `cf_cate_distribuicao.png` | Histograma da distribuição dos τ̂ |
| `cf_cate_por_regiao.png` | Boxplot do CATE por macrorregião |

---

## Resultados esperados

Os valores numéricos exatos dependem de versões dos pacotes, do
*seed* e da partição em árvores, mas os padrões substantivos
esperados são:

- **ATE *overlap-weighted* positivo** — municípios com alta exposição
  à orientação técnica apresentam, em média, produtividade
  agropecuária maior que os de baixa exposição, condicional aos
  controles. Tipicamente o ATE *overlap-weighted* fica entre o ATT
  (estimado sobre os tratados) e o ATC (estimado sobre os controles),
  com IC 95% que exclui zero.
- **Heterogeneidade detectável** — o teste de calibração rejeita a
  hipótese nula de τ(x) constante; os efeitos variam entre municípios.
- **Heterogeneidade por escala** — efeitos tendem a ser maiores
  (em log-pontos) para estabelecimentos com **escala intermediária**,
  consistente com a literatura agronômica sobre extensão rural.
- **Heterogeneidade regional** — Sul e Sudeste, com alta proporção
  de munis no grupo tratado (92% e 70%), tendem a mostrar efeitos
  positivos mais consistentes; Nordeste, com apenas 12% de munis
  tratados, exibe maior dispersão do CATE e suporte mais limitado
  da variável de tratamento — esse padrão é, ele próprio, parte do
  diagnóstico.
- **Variáveis mais importantes** para a heterogeneidade — escala
  produtiva (`log_area_media`, `log_pib_agropec`) e clima
  (`temperatura_media`, `precipitacao_anual`) tendem a dominar.

> **Atenção interpretativa.** A magnitude de τ̂ deve ser lida como
> *diferença esperada em log(produtividade agropecuária)* entre um
> município com alta exposição à orientação técnica e um similar com
> baixa exposição, **sob ignorabilidade condicional em X**. Veja
> também o caveat na seção [Modelo e método](#modelo-e-método).

---

## Solução de problemas

### Windows: `Rscript` não é reconhecido (PowerShell)

> *"O termo 'Rscript' não é reconhecido como nome de cmdlet..."*

O R está instalado mas não está no PATH. Use o caminho completo:

```powershell
& "C:\Program Files\R\R-4.4.2\bin\Rscript.exe" R\exemplo4.R dados\base_assistencia_tecnica.csv
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

### O pacote `grf` falha ao instalar (problema de compilação C++)

O `grf` tem código nativo e requer um compilador C++14 em algumas
plataformas. Em Windows, instale primeiro o
[Rtools](https://cran.r-project.org/bin/windows/Rtools/); em macOS,
instale as **Xcode Command Line Tools** (`xcode-select --install`); em
Linux, instale `build-essential` (Debian/Ubuntu).

### Aviso "Predictor variables have unusually large values..."

Pode aparecer quando colunas em escala bruta (PIB total, área em ha,
população) entram em X sem log. A versão atual do script já aplica log
nessas variáveis. Se ainda assim o aviso aparecer, ignore-o — não
afeta a consistência do estimador.

### Execução muito lenta

Reduza `num.trees = 2000` para `500` ou `1000`, ou desligue o
auto-tuning (`tune.parameters = "none"`). O ATE muda muito pouco;
a heterogeneidade fica menos suavizada.

---

## Como citar

Se você utilizar este material em pesquisa ou ensino, por favor cite o
capítulo do livro:

>
> (no prelo)
>

e as referências fundadoras do método:

> Wager, S., & Athey, S. (2018). Estimation and inference of
> heterogeneous treatment effects using random forests.
> *Journal of the American Statistical Association*, 113(523),
> 1228–1242.

> Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random
> forests. *The Annals of Statistics*, 47(2), 1148–1178.

E mencione este repositório:

> Repositório de apoio ao Exemplo 4 — Efeitos heterogêneos da
> assistência técnica agrícola sobre a produtividade agropecuária
> municipal via Causal Forest, cross-section 2017. Disponível em:
> `https://github.com/luizsatolo/livro-les-cap-ml (pasta `exemplo4-assistencia-tecnica-causal-forest/`)`

---

## Licença

- **Código**: MIT (ver arquivo [`LICENSE`](LICENSE)).
- **Dados**: Creative Commons Attribution 4.0 International (CC-BY-4.0),
  conforme as licenças das fontes (IBGE/Censo Agropecuário,
  ERA5/Copernicus, MapBiomas, SICOR/Bacen).

---

## Contato

Dúvidas, sugestões ou problemas de reprodução: abra uma *issue* neste
repositório.
