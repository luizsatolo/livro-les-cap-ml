#!/usr/bin/env Rscript

# =============================================================================
# Script:        exemplo1.R
# Capítulo:      Machine Learning em Avaliação de Impacto na Economia
# Exemplo:       1 - Previsão da produtividade da soja em municípios brasileiros
# Autor:         (preencher)
# Última revisão: 2025
#
# Objetivo
# -----------------------------------------------------------------------------
# Comparar três estratégias de modelagem para prever a produtividade da soja
# (kg/ha) em municípios brasileiros entre 2004 e 2021, utilizando variáveis
# climáticas (ERA5), edáficas (Mapbiomas), demográficas e econômicas (IBGE):
#
#   (1) LASSO puro              : regressão penalizada sobre todas as covariáveis
#   (2) FE Dinâmico             : efeitos fixos de município + lag da produtividade
#   (3) Híbrido (FE + LASSO)    : FE Dinâmico + LASSO sobre o resíduo do FE
#
# Estratégia metodológica
# -----------------------------------------------------------------------------
#   - O painel é filtrado para o período 2004-2021.
#   - Cria-se o lag de 1 ano da produtividade em kg/ha.
#   - A *amostra comum* é definida ANTES da estimação dos modelos
#     (de forma puramente determinística), garantindo comparabilidade.
#     Os três modelos são então estimados nessa mesma amostra na ordem
#     LASSO -> FE Dinâmico -> Híbrido.
#   - Split temporal: treino = anos <= 2018; teste = anos > 2018 (2019-2021).
#   - Imputação por mediana (calculada no treino) para covariáveis numéricas,
#     exceto o lag (observações sem lag são excluídas).
#   - Reprodutibilidade: set.seed(123) é executado uma vez no início do script
#     e novamente imediatamente antes de cada chamada de cv.glmnet, de modo
#     que cada LASSO recebe o mesmo estado do RNG independentemente da ordem
#     de execução.
#
# Execução
# -----------------------------------------------------------------------------
#   $ Rscript R/exemplo1.R dados/base_final.csv
#
# Se nenhum argumento for fornecido, o script tenta ler "dados/base_final.csv"
# a partir do diretório de trabalho atual.
#
# Saídas (em resultados_modelos/)
# -----------------------------------------------------------------------------
#   metricas_comparacao.csv          Métricas RMSE, MAE e R² por modelo
#   previsoes_comparacao.csv         Previsões municipais no conjunto de teste
#   coeficientes_lasso.csv           Coeficientes não nulos do LASSO puro
#   coeficientes_hibrido.csv         Coeficientes não nulos do LASSO do híbrido
#   observado_vs_previsto.png        Dispersão observado x previsto por modelo
#   serie_municipios_extremos.png    Série temporal: município de menor e maior
#                                    produtividade média e média geral
# =============================================================================


# =============================================================================
# 1. Pacotes
# =============================================================================

required_packages <- c(
  "readr",     # leitura de CSV
  "dplyr",     # manipulação de data frames
  "tidyr",     # pivots e drop_na
  "glmnet",    # LASSO via cv.glmnet
  "fixest",    # estimação de efeitos fixos (feols)
  "yardstick", # métricas de avaliação (RMSE, MAE, R²)
  "tibble",    # tibbles
  "ggplot2"    # gráficos
)

installed <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!(pkg %in% installed)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
}

set.seed(123)


# =============================================================================
# 2. Argumentos da linha de comando e diretório de saída
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
arquivo_base <- ifelse(length(args) >= 1, args[1], "dados/base_final.csv")

dir_resultados <- "resultados_modelos"
dir.create(dir_resultados, recursive = TRUE, showWarnings = FALSE)

message("Lendo base: ", arquivo_base)

if (!file.exists(arquivo_base)) {
  stop("Arquivo de dados não encontrado: ", arquivo_base)
}


# =============================================================================
# 3. Leitura e validação da base
# =============================================================================

base <- readr::read_csv(arquivo_base, show_col_types = FALSE)

vars_obrigatorias <- c("id_municipio", "ano", "produtividade_soja")
faltantes <- setdiff(vars_obrigatorias, names(base))

if (length(faltantes) > 0) {
  stop(
    "Variáveis obrigatórias ausentes na base: ",
    paste(faltantes, collapse = ", ")
  )
}


# =============================================================================
# 4. Preparação dos dados
# =============================================================================
# - Conversão de tipos
# - Filtro temporal (2004-2021)
# - Remoção de colunas textuais não identificadoras
# - Criação do lag de 1 ano da produtividade
# =============================================================================

base <- base |>
  dplyr::mutate(
    id_municipio  = as.character(id_municipio),
    ano           = as.integer(ano),
    produtividade = as.numeric(produtividade_soja)
  ) |>
  dplyr::filter(
    !is.na(produtividade),
    produtividade > 0,
    ano >= 2004,
    ano <= 2021
  ) |>
  dplyr::arrange(id_municipio, ano)

# Remove colunas textuais (exceto o identificador do município)
cols_texto <- names(base)[sapply(base, is.character)]
cols_texto_remover <- setdiff(cols_texto, "id_municipio")

base <- base |>
  dplyr::select(-dplyr::any_of(cols_texto_remover)) |>
  dplyr::select(where(~ !all(is.na(.x))))

# Lag da produtividade (dentro de cada município, após o filtro temporal)
base <- base |>
  dplyr::group_by(id_municipio) |>
  dplyr::arrange(ano, .by_group = TRUE) |>
  dplyr::mutate(
    produtividade_lag = dplyr::lag(produtividade, 1)
  ) |>
  dplyr::ungroup()

message(
  "Período analisado: ",
  min(base$ano, na.rm = TRUE), "-", max(base$ano, na.rm = TRUE)
)


# =============================================================================
# 5. Split temporal (treino/teste)
# =============================================================================

ano_corte <- 2018

train <- base |> dplyr::filter(ano <= ano_corte)
test  <- base |> dplyr::filter(ano >  ano_corte)

# Mantém no teste apenas municípios presentes no treino
test <- test |>
  dplyr::filter(id_municipio %in% unique(train$id_municipio))

message("Observações brutas no treino: ", nrow(train))
message("Observações brutas no teste : ", nrow(test))


# =============================================================================
# 6. Seleção das covariáveis numéricas
# =============================================================================

vars_excluir <- c(
  "id_municipio",
  "ano",
  "produtividade_soja",
  "produtividade"
)

vars <- setdiff(names(train), vars_excluir)
vars <- vars[sapply(train[vars], is.numeric)]

# O lag deve sempre estar nas covariáveis
if (!("produtividade_lag" %in% vars)) {
  stop("produtividade_lag não foi criada corretamente.")
}

# Remove covariáveis sem variação no treino
vars_const <- vars[
  sapply(train[vars], function(x) length(unique(x[!is.na(x)])) <= 1)
]

if (length(vars_const) > 0) {
  message(
    "Removendo covariáveis constantes no treino: ",
    paste(vars_const, collapse = ", ")
  )
  vars <- setdiff(vars, vars_const)
}


# =============================================================================
# 7. Imputação por mediana (estimada no treino)
# =============================================================================
# Não imputamos produtividade_lag: observações sem lag devem sair da amostra
# comum. As demais covariáveis numéricas podem ser imputadas para reduzir
# perda amostral.
# =============================================================================

vars_imputar <- setdiff(vars, "produtividade_lag")

medianas <- train |>
  dplyr::summarise(
    dplyr::across(
      dplyr::all_of(vars_imputar),
      ~ median(.x, na.rm = TRUE)
    )
  )

vars_sem_mediana <- names(medianas)[
  sapply(medianas, function(x) {
    is.na(x[[1]]) || is.nan(x[[1]]) || !is.finite(x[[1]])
  })
]

if (length(vars_sem_mediana) > 0) {
  message(
    "Removendo variáveis sem mediana no treino: ",
    paste(vars_sem_mediana, collapse = ", ")
  )

  vars         <- setdiff(vars,         vars_sem_mediana)
  vars_imputar <- setdiff(vars_imputar, vars_sem_mediana)

  train <- train |> dplyr::select(-dplyr::all_of(vars_sem_mediana))
  test  <- test  |> dplyr::select(-dplyr::all_of(vars_sem_mediana))

  medianas <- train |>
    dplyr::summarise(
      dplyr::across(
        dplyr::all_of(vars_imputar),
        ~ median(.x, na.rm = TRUE)
      )
    )
}

imputar_mediana <- function(df, medianas, vars_imputar) {
  for (v in vars_imputar) {
    if (v %in% names(df)) {
      med <- medianas[[v]][[1]]
      sub <- is.na(df[[v]]) | is.nan(df[[v]]) | !is.finite(df[[v]])
      df[[v]][sub] <- med
    }
  }
  df
}

train <- imputar_mediana(train, medianas, vars_imputar)
test  <- imputar_mediana(test,  medianas, vars_imputar)

# Troca Inf/-Inf/NaN remanescentes por NA em todas as numéricas
limpar_nao_finitos <- function(df) {
  df |>
    dplyr::mutate(
      dplyr::across(
        where(is.numeric),
        ~ ifelse(is.finite(.x), .x, NA_real_)
      )
    )
}

train <- limpar_nao_finitos(train)
test  <- limpar_nao_finitos(test)

message("Número de covariáveis numéricas usadas: ", length(vars))


# =============================================================================
# 8. Funções auxiliares
# =============================================================================

# Linha de métricas para um modelo
metricas_linha <- function(nome, truth, estimate) {
  tibble::tibble(
    modelo = nome,
    RMSE   = yardstick::rmse_vec(truth = truth, estimate = estimate),
    MAE    = yardstick::mae_vec(truth = truth, estimate = estimate),
    R2     = yardstick::rsq_vec(truth = truth, estimate = estimate)
  )
}

# Extrai coeficientes não nulos de um cv.glmnet em um lambda dado
extract_nonzero <- function(modelo, s, nome) {
  coefs <- coef(modelo, s = s)
  data.frame(
    modelo      = nome,
    variavel    = rownames(as.matrix(coefs)),
    coeficiente = as.numeric(coefs)
  ) |>
    dplyr::filter(coeficiente != 0) |>
    dplyr::arrange(dplyr::desc(abs(coeficiente)))
}

# Cria matriz de design garantindo amostra completa, devolvendo X e df alinhados
criar_matriz <- function(formula, df, nome) {
  mf   <- model.frame(formula, data = df, na.action = na.pass)
  keep <- complete.cases(mf)

  if (sum(!keep) > 0) {
    message(
      "Aviso: ", nome, " removeu ", sum(!keep),
      " linhas por NA no model.frame."
    )
  }

  df_clean <- df[keep, , drop = FALSE]
  X        <- model.matrix(formula, data = df_clean)

  list(
    X  = X[, -1, drop = FALSE],   # remove intercepto
    df = df_clean
  )
}


# =============================================================================
# 9. Definição da amostra comum (puramente determinística)
# =============================================================================
# A amostra comum é fixada ANTES da estimação dos modelos para garantir que
# as métricas dos três modelos sejam diretamente comparáveis e para que a
# ordem de estimação (LASSO -> FE -> Híbrido) não interfira nos resultados.
#
# Passos:
#   9.1 drop_na nas variáveis usadas (produtividade, lag e covariáveis).
#   9.2 Mantém no treino apenas municípios com pelo menos 2 observações
#       (condição necessária para o FE).
#   9.3 Mantém no teste apenas municípios presentes no treino.
#   9.4 Constrói a matriz de design comum X (sobre 'produtividade ~ vars').
#       A mesma matriz será reutilizada pelo LASSO puro e pelo LASSO do
#       Híbrido (ambos compartilham o mesmo X), eliminando quaisquer linhas
#       remanescentes com NA no model.frame e colunas constantes.
# =============================================================================

# 9.1 drop_na nas variáveis necessárias
train_common <- train |>
  dplyr::select(id_municipio, ano, produtividade, produtividade_lag,
                dplyr::all_of(vars)) |>
  tidyr::drop_na()

test_common <- test |>
  dplyr::select(id_municipio, ano, produtividade, produtividade_lag,
                dplyr::all_of(vars)) |>
  tidyr::drop_na()

# 9.2 Remove municípios singleton no treino (impossíveis para FE)
ids_validos <- train_common |>
  dplyr::count(id_municipio) |>
  dplyr::filter(n >= 2) |>
  dplyr::pull(id_municipio)

train_common <- train_common |> dplyr::filter(id_municipio %in% ids_validos)
test_common  <- test_common  |> dplyr::filter(id_municipio %in% ids_validos)

# 9.3 Garante municípios do teste presentes no treino
test_common <- test_common |>
  dplyr::filter(id_municipio %in% unique(train_common$id_municipio))

if (nrow(train_common) == 0 || nrow(test_common) == 0) {
  stop("Amostra comum vazia após limpeza.")
}

# 9.4 Constrói a matriz de design comum (X) sobre 'produtividade ~ vars'
form_X <- as.formula(
  paste("produtividade ~", paste(vars, collapse = " + "))
)

mat_train <- criar_matriz(form_X, train_common, "Amostra comum (treino)")
mat_test  <- criar_matriz(form_X, test_common,  "Amostra comum (teste)")

train_common <- mat_train$df
test_common  <- mat_test$df

X_train <- mat_train$X
X_test  <- mat_test$X

# Remove colunas constantes no treino (por segurança)
cols_var <- apply(X_train, 2, function(x) length(unique(x[!is.na(x)])) > 1)
X_train  <- X_train[, cols_var, drop = FALSE]
X_test   <- X_test[,  cols_var, drop = FALSE]

if (ncol(X_train) < 2) {
  stop("Há menos de 2 covariáveis válidas para glmnet na amostra comum.")
}

if (nrow(X_train) != nrow(train_common)) {
  stop("Desalinhamento: X_train e train_common têm tamanhos diferentes.")
}

message("Amostra comum (treino): ", nrow(train_common), " obs")
message("Amostra comum (teste) : ", nrow(test_common), " obs")


# =============================================================================
# 10. Modelo 1 — LASSO puro
# =============================================================================
# Penaliza todas as covariáveis (inclusive o lag) sobre a produtividade
# em nível.
# =============================================================================

y_train <- train_common$produtividade

set.seed(123)
lasso <- glmnet::cv.glmnet(
  x           = X_train,
  y           = y_train,
  alpha       = 1,
  standardize = TRUE,
  nfolds      = 10
)

pred_lasso <- as.numeric(
  predict(lasso, newx = X_test, s = "lambda.1se")
)


# =============================================================================
# 11. Modelo 2 — FE Dinâmico
# =============================================================================
# Efeitos fixos de município com lag da produtividade como única covariável.
# =============================================================================

fe_model <- fixest::feols(
  produtividade ~ produtividade_lag | id_municipio,
  data  = train_common,
  warn  = TRUE,
  notes = FALSE
)

# Previsões no treino (para uso no híbrido) e no teste
fe_fit_train <- as.numeric(predict(fe_model, newdata = train_common))
pred_fe      <- as.numeric(predict(fe_model, newdata = test_common))

# Assertiva: dado o filtro de singletons e o alinhamento de municípios entre
# treino e teste, o FE não deve produzir NAs. Esta verificação garante que a
# amostra comum permaneça idêntica à que seria obtida se o híbrido fosse
# estimado primeiro.
if (any(is.na(fe_fit_train)) || any(is.na(pred_fe))) {
  stop("FE retornou NAs em algum município da amostra comum; ",
       "revisar filtros de id_municipio.")
}

if (length(pred_fe) != nrow(test_common)) {
  stop("FE não retornou predições alinhadas à amostra comum.")
}


# =============================================================================
# 12. Modelo 3 — Híbrido FE + LASSO
# =============================================================================
# Calcula o resíduo do FE no treino e ajusta um LASSO sobre esse resíduo,
# usando a mesma matriz de design X da amostra comum. A previsão final é
# o ajuste do FE no teste somado à previsão LASSO do resíduo.
# =============================================================================

resid_train <- train_common$produtividade - fe_fit_train

set.seed(123)
h_lasso <- glmnet::cv.glmnet(
  x           = X_train,
  y           = resid_train,
  alpha       = 1,
  standardize = TRUE,
  nfolds      = 10
)

pred_resid   <- as.numeric(
  predict(h_lasso, newx = X_test, s = "lambda.1se")
)
pred_hibrido <- pred_fe + pred_resid


# =============================================================================
# 13. Métricas de avaliação no conjunto de teste
# =============================================================================

y <- test_common$produtividade

metricas <- dplyr::bind_rows(
  metricas_linha("LASSO",              y, pred_lasso),
  metricas_linha("FE Dinâmico",        y, pred_fe),
  metricas_linha("Híbrido FE + LASSO", y, pred_hibrido)
)

print(metricas)


# =============================================================================
# 14. Salva métricas, previsões e coeficientes
# =============================================================================

readr::write_csv(
  metricas,
  file.path(dir_resultados, "metricas_comparacao.csv")
)

previsoes <- tibble::tibble(
  id_municipio       = test_common$id_municipio,
  ano                = test_common$ano,
  observado_kg_ha    = y,
  pred_lasso_kg_ha   = pred_lasso,
  pred_fe_kg_ha      = pred_fe,
  pred_hibrido_kg_ha = pred_hibrido
)

readr::write_csv(
  previsoes,
  file.path(dir_resultados, "previsoes_comparacao.csv")
)

coef_lasso   <- extract_nonzero(lasso,   "lambda.1se", "LASSO")
coef_hibrido <- extract_nonzero(h_lasso, "lambda.1se",
                                "Híbrido FE + LASSO - resíduo FE")

readr::write_csv(
  coef_lasso,
  file.path(dir_resultados, "coeficientes_lasso.csv")
)

readr::write_csv(
  coef_hibrido,
  file.path(dir_resultados, "coeficientes_hibrido.csv")
)


# =============================================================================
# 15. Gráfico 1: observado vs. previsto (dispersão por modelo)
# =============================================================================

grafico_disp <- previsoes |>
  dplyr::select(
    observado_kg_ha,
    pred_lasso_kg_ha,
    pred_fe_kg_ha,
    pred_hibrido_kg_ha
  ) |>
  tidyr::pivot_longer(
    cols      = dplyr::starts_with("pred_"),
    names_to  = "modelo",
    values_to = "previsto"
  ) |>
  dplyr::mutate(
    modelo = dplyr::recode(
      modelo,
      pred_lasso_kg_ha   = "LASSO",
      pred_fe_kg_ha      = "FE Dinâmico",
      pred_hibrido_kg_ha = "Híbrido FE + LASSO"
    )
  ) |>
  ggplot2::ggplot(ggplot2::aes(x = observado_kg_ha, y = previsto)) +
  ggplot2::geom_point(alpha = 0.25) +
  ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  ggplot2::facet_wrap(~ modelo) +
  ggplot2::labs(
    x     = "Produtividade observada (kg/ha)",
    y     = "Produtividade prevista (kg/ha)",
    title = "Observado vs. previsto"
  ) +
  ggplot2::theme_minimal()

ggplot2::ggsave(
  filename = file.path(dir_resultados, "observado_vs_previsto.png"),
  plot     = grafico_disp,
  width    = 11,
  height   = 5.5,
  dpi      = 300
)


# =============================================================================
# 16. Gráfico 2: série temporal nos extremos e na média municipal
# =============================================================================
# Identifica os municípios de menor e maior produtividade média no teste e
# plota, ao lado da média geral, as séries temporais de observado e previsto.
# =============================================================================

medias_mun <- previsoes |>
  dplyr::group_by(id_municipio) |>
  dplyr::summarise(
    prod_media = mean(observado_kg_ha, na.rm = TRUE),
    .groups    = "drop"
  )

mun_min <- medias_mun |>
  dplyr::slice_min(prod_media, n = 1, with_ties = FALSE) |>
  dplyr::pull(id_municipio)

mun_max <- medias_mun |>
  dplyr::slice_max(prod_media, n = 1, with_ties = FALSE) |>
  dplyr::pull(id_municipio)

# Média municipal por ano no teste
prev_media <- previsoes |>
  dplyr::group_by(ano) |>
  dplyr::summarise(
    observado_kg_ha    = mean(observado_kg_ha,    na.rm = TRUE),
    pred_lasso_kg_ha   = mean(pred_lasso_kg_ha,   na.rm = TRUE),
    pred_fe_kg_ha      = mean(pred_fe_kg_ha,      na.rm = TRUE),
    pred_hibrido_kg_ha = mean(pred_hibrido_kg_ha, na.rm = TRUE),
    .groups            = "drop"
  ) |>
  dplyr::mutate(painel = "Média dos municípios")

# Séries dos municípios extremos
prev_extremos <- previsoes |>
  dplyr::filter(id_municipio %in% c(mun_min, mun_max)) |>
  dplyr::mutate(
    painel = dplyr::case_when(
      id_municipio == mun_min ~ paste0("Menor produtividade média: ", mun_min),
      id_municipio == mun_max ~ paste0("Maior produtividade média: ", mun_max)
    )
  ) |>
  dplyr::select(
    painel, ano,
    observado_kg_ha,
    pred_lasso_kg_ha,
    pred_fe_kg_ha,
    pred_hibrido_kg_ha
  )

plot_df <- dplyr::bind_rows(prev_extremos, prev_media) |>
  dplyr::mutate(
    painel = factor(
      painel,
      levels = c(
        paste0("Menor produtividade média: ", mun_min),
        "Média dos municípios",
        paste0("Maior produtividade média: ", mun_max)
      )
    )
  ) |>
  tidyr::pivot_longer(
    cols = c(
      observado_kg_ha,
      pred_lasso_kg_ha,
      pred_fe_kg_ha,
      pred_hibrido_kg_ha
    ),
    names_to  = "serie",
    values_to = "valor"
  ) |>
  dplyr::mutate(
    serie = dplyr::recode(
      serie,
      observado_kg_ha    = "Observado",
      pred_lasso_kg_ha   = "LASSO",
      pred_fe_kg_ha      = "FE Dinâmico",
      pred_hibrido_kg_ha = "Híbrido FE + LASSO"
    )
  )

grafico_serie <- ggplot2::ggplot(
    plot_df,
    ggplot2::aes(x = ano, y = valor, color = serie)
  ) +
  ggplot2::geom_line(linewidth = 1.1) +
  ggplot2::geom_point(size = 2) +
  ggplot2::facet_wrap(~ painel, nrow = 1, drop = FALSE) +
  ggplot2::scale_x_continuous(
    breaks       = c(2019, 2020, 2021),
    minor_breaks = NULL
  ) +
  ggplot2::labs(
    x     = "Ano",
    y     = "Produtividade da soja (kg/ha)",
    color = ""
  ) +
  ggplot2::theme_minimal(base_size = 13) +
  ggplot2::theme(legend.position = "bottom")

ggplot2::ggsave(
  filename = file.path(dir_resultados, "serie_municipios_extremos.png"),
  plot     = grafico_serie,
  width    = 12,
  height   = 5,
  dpi      = 300
)


# =============================================================================
# 17. Fim
# =============================================================================

message("Resultados salvos em: ", dir_resultados)
