#!/usr/bin/env Rscript

# =============================================================================
# Script:        exemplo3.R
# Capítulo:      Machine Learning em Avaliação de Impacto na Economia
# Exemplo:       3 - Efeito do PRONAF sobre a produtividade do milho municipal
#                    via Double/Debiased Machine Learning (DML) em painel
# Autor:         (preencher)
# Última revisão: 2025
#
# Objetivo
# -----------------------------------------------------------------------------
# Estimar o efeito causal da intensidade de exposição ao PRONAF (Programa
# Nacional de Fortalecimento da Agricultura Familiar) sobre a produtividade
# do milho em municípios brasileiros entre 2013 e 2021, controlando por um
# vetor rico de características climáticas (ERA5), edáficas (MapBiomas),
# demográficas e econômicas (IBGE) por meio do Double/Debiased Machine
# Learning de Chernozhukov et al. (2018).
#
# Trata-se de um PAINEL anual 2013-2021 (município × ano).
#
# Modelo parcialmente linear (Robinson, 1988):
#
#   Y_it = θ D_it + g(X_it) + u_it
#   D_it =         m(X_it) + v_it
#
# em que θ é o efeito causal de interesse, g e m são funções desconhecidas
# das covariáveis estimadas por ML (LASSO e Random Forest) com cross-fitting
# (K-fold). A inferência usa erros-padrão robustos a agrupamento
# (cluster) por município, via DoubleMLClusterData. O X inclui *dummies*
# de macrorregião e de ano, absorvendo heterogeneidade regional e choques
# temporais comuns.
#
# Variáveis
# -----------------------------------------------------------------------------
#   Y = log(produtividade_milho)           (PAM/IBGE, kg/ha em log natural)
#   D = log(1 + valor_pronaf)              (SICOR/Bacen, valor anual em R$)
#   X = clima (ERA5) + solo (MapBiomas) + socioeconômico (IBGE)
#       + 4 dummies de macrorregião + 8 dummies de ano
#
# Execução
# -----------------------------------------------------------------------------
#   $ Rscript R/exemplo3.R dados/base_pronaf.csv
#
# Saídas (em resultados_modelos/)
# -----------------------------------------------------------------------------
#   base_dml_processada.csv          Base efetivamente usada no DML
#   dml_resultados.csv               Estimativas, SE clustered, IC e p-valores
#   dml_residuos.csv                 Resíduos do partialling-out
#   dml_forest_plot.png              Comparação de θ̂ entre LASSO e RF
#   importancia_variaveis_rf.csv     Importância nas equações nuisance (RF)
# =============================================================================


# =============================================================================
# 1. Pacotes
# =============================================================================

required_packages <- c(
  "readr",         # leitura de CSV
  "dplyr",         # manipulação de data frames
  "tidyr",         # drop_na
  "tibble",        # tibbles
  "ggplot2",       # gráficos
  "DoubleML",      # implementação do DML (incl. DoubleMLClusterData)
  "mlr3",          # framework de ML
  "mlr3learners",  # aprendizes (LASSO, RF) para mlr3
  "glmnet",        # backend do LASSO (regr.cv_glmnet)
  "ranger"         # backend do Random Forest (regr.ranger)
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

if ("lgr" %in% installed) {
  lgr::get_logger("mlr3")$set_threshold("warn")
}

set.seed(123)


# =============================================================================
# 2. Argumentos da linha de comando e diretório de saída
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
arquivo_base <- ifelse(length(args) >= 1, args[1], "dados/base_pronaf.csv")

dir_resultados <- "resultados_modelos"
dir.create(dir_resultados, recursive = TRUE, showWarnings = FALSE)

message("Lendo base: ", arquivo_base)

if (!file.exists(arquivo_base)) {
  stop("Arquivo de dados não encontrado: ", arquivo_base)
}


# =============================================================================
# 3. Leitura e validação da base
# =============================================================================

base <- readr::read_csv(
  arquivo_base,
  show_col_types = FALSE,
  col_types = readr::cols(
    id_municipio = readr::col_character(),
    ano          = readr::col_integer(),
    regiao       = readr::col_character(),
    .default     = readr::col_double()
  )
)

vars_obrigatorias <- c(
  "id_municipio", "ano", "produtividade_milho", "valor_pronaf",
  "regiao",
  "temperatura_media", "temperatura_maxima", "dias_quentes",
  "precipitacao_anual", "precipitacao_extrema", "dias_secos",
  "carbono_solo", "teor_argila", "teor_areia",
  "pib_total", "pib_agropecuaria", "pib_industria", "pib_servicos",
  "populacao_total", "area_municipio", "pib_percapita",
  "densidade_demografica"
)

faltantes <- setdiff(vars_obrigatorias, names(base))
if (length(faltantes) > 0) {
  stop("Variáveis ausentes na base: ", paste(faltantes, collapse = ", "))
}

message("Leituras: ", nrow(base), " obs (município × ano)")
message("Período: ", min(base$ano), "-", max(base$ano))
message("Municípios: ", length(unique(base$id_municipio)))


# =============================================================================
# 4. Preparação dos dados
# =============================================================================
# - Filtros mínimos (produtividade observada e positiva, valor PRONAF >= 0).
# - Transformações log em Y (produtividade) e D (PRONAF).
# - log nas variáveis socioeconômicas de escala.
# - Dummies de macrorregião (Norte = referência) e de ano (2013 = referência).
# =============================================================================

log_pos <- function(z) log(pmax(z, 1))

base <- base |>
  dplyr::filter(
    !is.na(produtividade_milho),
    produtividade_milho > 0,
    !is.na(valor_pronaf),
    valor_pronaf >= 0
  ) |>
  dplyr::mutate(
    log_produtividade = log(produtividade_milho),
    log_pronaf        = log1p(valor_pronaf),
    log_pib_total     = log_pos(pib_total),
    log_pib_agropec   = log_pos(pib_agropecuaria),
    log_pib_industria = log_pos(pib_industria),
    log_pib_servicos  = log_pos(pib_servicos),
    log_populacao     = log_pos(populacao_total),
    log_area          = log_pos(area_municipio),
    log_pib_percapita = log_pos(pib_percapita),
    # ID numérico para uso como variável de cluster
    id_mun_num        = as.integer(id_municipio)
  )

# Dummies de região (Norte como referência)
base <- base |>
  dplyr::mutate(
    regiao_NE  = as.integer(regiao == "Nordeste"),
    regiao_SE  = as.integer(regiao == "Sudeste"),
    regiao_SUL = as.integer(regiao == "Sul"),
    regiao_CO  = as.integer(regiao == "Centro-Oeste")
  )

# Dummies de ano (2013 como referência)
anos_dummy <- setdiff(sort(unique(base$ano)), 2013)
for (a in anos_dummy) {
  base[[paste0("ano_", a)]] <- as.integer(base$ano == a)
}


# =============================================================================
# 5. Especificação das variáveis Y, D e X
# =============================================================================

vars_clima <- c(
  "temperatura_media", "temperatura_maxima",
  "dias_quentes", "precipitacao_anual",
  "precipitacao_extrema", "dias_secos"
)

vars_solo <- c("carbono_solo", "teor_argila", "teor_areia")

vars_socio <- c(
  "log_pib_total", "log_pib_agropec",
  "log_pib_industria", "log_pib_servicos",
  "log_populacao", "log_area",
  "log_pib_percapita", "densidade_demografica"
)

vars_regiao <- c("regiao_NE", "regiao_SE", "regiao_SUL", "regiao_CO")
vars_ano    <- paste0("ano_", anos_dummy)

vars_x <- c(vars_clima, vars_solo, vars_socio, vars_regiao, vars_ano)

dados_dml <- base |>
  dplyr::select(
    id_mun_num,
    y = log_produtividade,
    d = log_pronaf,
    dplyr::all_of(vars_x)
  ) |>
  tidyr::drop_na() |>
  as.data.frame()

stopifnot(all(sapply(dados_dml[, c("y", "d", vars_x)], is.numeric)))
stopifnot(!any(is.na(dados_dml)))

message("Linhas usadas no DML: ", nrow(dados_dml))
message("Número de covariáveis X: ", length(vars_x),
        " (", length(vars_clima), " clima, ",
        length(vars_solo), " solo, ",
        length(vars_socio), " socio, ",
        length(vars_regiao), " região, ",
        length(vars_ano), " ano)")

readr::write_csv(
  dados_dml,
  file.path(dir_resultados, "base_dml_processada.csv")
)


# =============================================================================
# 6. Construção do objeto DoubleMLClusterData
# =============================================================================
# Usamos a versão "cluster" do DoubleML, que produz inferência robusta a
# correlação serial dentro do município (cluster = id_municipio).
# =============================================================================

dml_data <- DoubleML::DoubleMLClusterData$new(
  data         = dados_dml,
  y_col        = "y",
  d_cols       = "d",
  cluster_cols = "id_mun_num",
  x_cols       = vars_x
)

print(dml_data)


# =============================================================================
# 7. DML com LASSO como aprendiz auxiliar
# =============================================================================

message("Estimando DML com LASSO...")

ml_l_lasso <- mlr3::lrn("regr.cv_glmnet", s = "lambda.min", nfolds = 5)
ml_m_lasso <- mlr3::lrn("regr.cv_glmnet", s = "lambda.min", nfolds = 5)

set.seed(123)
dml_lasso <- DoubleML::DoubleMLPLR$new(
  data    = dml_data,
  ml_l    = ml_l_lasso,
  ml_m    = ml_m_lasso,
  n_folds = 5,
  score   = "partialling out"
)

dml_lasso$fit(store_predictions = TRUE, store_models = TRUE)
print(dml_lasso$summary())


# =============================================================================
# 8. DML com Random Forest como aprendiz auxiliar
# =============================================================================

message("Estimando DML com Random Forest...")

p_x     <- length(vars_x)
mtry_rf <- max(1, floor(sqrt(p_x)))

ml_l_rf <- mlr3::lrn(
  "regr.ranger",
  num.trees     = 500,
  mtry          = mtry_rf,
  min.node.size = 5,
  importance    = "impurity"
)

ml_m_rf <- mlr3::lrn(
  "regr.ranger",
  num.trees     = 500,
  mtry          = mtry_rf,
  min.node.size = 5,
  importance    = "impurity"
)

set.seed(123)
dml_rf <- DoubleML::DoubleMLPLR$new(
  data    = dml_data,
  ml_l    = ml_l_rf,
  ml_m    = ml_m_rf,
  n_folds = 5,
  score   = "partialling out"
)

dml_rf$fit(store_predictions = TRUE, store_models = TRUE)
print(dml_rf$summary())


# =============================================================================
# 9. Tabela comparativa de resultados
# =============================================================================

linha_resultado <- function(obj, nome) {
  est <- as.numeric(obj$coef)
  se  <- as.numeric(obj$se)
  tibble::tibble(
    modelo  = nome,
    theta   = est,
    se      = se,
    ci_inf  = est - 1.96 * se,
    ci_sup  = est + 1.96 * se,
    z       = est / se,
    p_valor = 2 * (1 - pnorm(abs(est / se)))
  )
}

resultados <- dplyr::bind_rows(
  linha_resultado(dml_lasso, "DML + LASSO"),
  linha_resultado(dml_rf,    "DML + Random Forest")
)

print(resultados)

readr::write_csv(
  resultados,
  file.path(dir_resultados, "dml_resultados.csv")
)


# =============================================================================
# 10. Forest plot das estimativas
# =============================================================================

grafico_dml <- ggplot2::ggplot(
    resultados,
    ggplot2::aes(x = theta, y = modelo, xmin = ci_inf, xmax = ci_sup)
  ) +
  ggplot2::geom_vline(xintercept = 0, linetype = 2, linewidth = 0.5) +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_errorbar(width = 0.15, orientation = "y") +
  ggplot2::labs(
    title    = "Efeito do PRONAF na produtividade do milho municipal",
    subtitle = "DML / modelo parcialmente linear, painel 2013-2021, SE clustered por município",
    x        = "θ̂ — variação esperada em log(produtividade) por unidade de log(1 + PRONAF) — IC95%",
    y        = ""
  ) +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::theme(
    panel.background = ggplot2::element_rect(fill = "gray95", color = NA),
    panel.grid.major = ggplot2::element_line(color = "white"),
    panel.grid.minor = ggplot2::element_blank()
  )

ggplot2::ggsave(
  filename = file.path(dir_resultados, "dml_forest_plot.png"),
  plot     = grafico_dml,
  width    = 10,
  height   = 4,
  dpi      = 300
)


# =============================================================================
# 11. Resíduos do partialling-out — diagnóstico
# =============================================================================
# Em PLR com score "partialling out":
#   ĝ(X) é estimada por ml_l  (regressão de Y em X)
#   m̂(X) é estimada por ml_m  (regressão de D em X)
# Os resíduos são (Y - ĝ(X)) e (D - m̂(X)).
#
# Diferentes versões do DoubleML guardam as predições em formatos distintos
# ($predictions$ml_l pode ser vetor, array 3D (n_obs × n_rep × n_treat),
# ou ainda uma lista nominal indexada pela coluna de tratamento).
# A função auxiliar abaixo lida com essas variações.
# =============================================================================

n_obs <- nrow(dados_dml)

extrair_predicao <- function(dml_obj, learner, d_col = "d", n_obs) {
  preds <- dml_obj$predictions
  if (is.null(preds) || length(preds) == 0) return(rep(NA_real_, n_obs))
  alvo <- preds[[learner]]
  # Caso a estrutura seja indexada pelo nome da coluna de tratamento
  if (is.null(alvo) && !is.null(preds[[d_col]])) {
    alvo <- preds[[d_col]][[learner]]
  }
  if (is.null(alvo) || length(alvo) == 0) return(rep(NA_real_, n_obs))
  if (is.array(alvo) && length(dim(alvo)) == 3) {
    # shape (n_obs, n_rep, n_treat) -> primeira rep, primeiro tratamento
    return(as.numeric(alvo[, 1, 1]))
  }
  vec <- as.numeric(alvo)
  if (length(vec) != n_obs) return(rep(NA_real_, n_obs))
  vec
}

pred_y_lasso <- extrair_predicao(dml_lasso, "ml_l", "d", n_obs)
pred_d_lasso <- extrair_predicao(dml_lasso, "ml_m", "d", n_obs)
pred_y_rf    <- extrair_predicao(dml_rf,    "ml_l", "d", n_obs)
pred_d_rf    <- extrair_predicao(dml_rf,    "ml_m", "d", n_obs)

preds_ok <- !all(is.na(pred_y_lasso)) &&
            !all(is.na(pred_d_lasso)) &&
            !all(is.na(pred_y_rf))    &&
            !all(is.na(pred_d_rf))

if (preds_ok) {
  residuos <- tibble::tibble(
    id_mun_num   = dados_dml$id_mun_num,
    y            = dados_dml$y,
    d            = dados_dml$d,
    pred_y_lasso = pred_y_lasso,
    pred_d_lasso = pred_d_lasso,
    pred_y_rf    = pred_y_rf,
    pred_d_rf    = pred_d_rf
  ) |>
    dplyr::mutate(
      res_y_lasso = y - pred_y_lasso,
      res_d_lasso = d - pred_d_lasso,
      res_y_rf    = y - pred_y_rf,
      res_d_rf    = d - pred_d_rf
    )

  readr::write_csv(
    residuos,
    file.path(dir_resultados, "dml_residuos.csv")
  )
  message("Resíduos do partialling-out salvos.")
} else {
  message(
    "Aviso: a versão instalada do pacote DoubleML não disponibilizou as ",
    "predições em $predictions; a tabela de resíduos não foi gravada. ",
    "Isso não afeta as estimativas pontuais nem os erros-padrão."
  )
}


# =============================================================================
# 12. Importância das variáveis nas equações auxiliares (Random Forest)
# =============================================================================
# Em vez de extrair as importâncias dos K x R modelos cross-fitted internos
# do DoubleML (cuja estrutura em $models varia entre versões do pacote),
# estimamos dois Random Forests diretamente em toda a amostra de DML —
# um para Y ~ X e outro para D ~ X — usando o mesmo hiperparâmetro
# (num.trees, mtry, min.node.size) das equações auxiliares. As importâncias
# resultantes são qualitativamente equivalentes às médias entre folds e mais
# robustas para diagnóstico.
# =============================================================================

message("Estimando importância de variáveis (Random Forest auxiliar)...")

set.seed(123)
rf_y_aux <- ranger::ranger(
  formula       = y ~ .,
  data          = dados_dml[, c("y", vars_x)],
  num.trees     = 500,
  mtry          = mtry_rf,
  min.node.size = 5,
  importance    = "impurity"
)

set.seed(123)
rf_d_aux <- ranger::ranger(
  formula       = d ~ .,
  data          = dados_dml[, c("d", vars_x)],
  num.trees     = 500,
  mtry          = mtry_rf,
  min.node.size = 5,
  importance    = "impurity"
)

importancia <- dplyr::bind_rows(
  tibble::tibble(
    equacao     = "Y ~ X (g)",
    variavel    = names(rf_y_aux$variable.importance),
    importancia = as.numeric(rf_y_aux$variable.importance)
  ),
  tibble::tibble(
    equacao     = "D ~ X (m)",
    variavel    = names(rf_d_aux$variable.importance),
    importancia = as.numeric(rf_d_aux$variable.importance)
  )
) |>
  dplyr::arrange(equacao, dplyr::desc(importancia)) |>
  dplyr::select(equacao, variavel, importancia)

readr::write_csv(
  importancia,
  file.path(dir_resultados, "importancia_variaveis_rf.csv")
)

message("Importancia de variaveis salva: ", nrow(importancia), " linhas.")


# =============================================================================
# 13. Fim
# =============================================================================

message("Resultados salvos em: ", dir_resultados)
