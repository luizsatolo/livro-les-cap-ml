#!/usr/bin/env Rscript

# =============================================================================
# Script:        exemplo4.R
# Capítulo:      Machine Learning em Avaliação de Impacto na Economia
# Exemplo:       4 - Efeitos heterogêneos da assistência técnica agrícola
#                    sobre a produtividade agropecuária municipal via
#                    Causal Forest (Wager & Athey, 2018)
# Autor:         (preencher)
# Última revisão: 2025
#
# Objetivo
# -----------------------------------------------------------------------------
# Estimar efeitos heterogêneos do recebimento de orientação/assistência técnica
# sobre o valor da produção agropecuária por hectare em municípios brasileiros
# em 2017, usando o estimador Causal Forest (pacote grf). O foco do exemplo é
# o CATE (Conditional Average Treatment Effect) e a identificação dos
# atributos municipais que mais explicam a heterogeneidade.
#
# Modelo (Wager e Athey, 2018; Athey, Tibshirani e Wager, 2019)
# -----------------------------------------------------------------------------
#   Y_i  = m(X_i) + W_i · τ(X_i) + ε_i
#   E[ε_i | X_i, W_i] = 0
#
# Em que:
#   Y_i = log(produtividade agropecuária)
#   W_i = 1 se o município está acima da mediana nacional na proporção de
#         estabelecimentos com orientação técnica recebida (Censo Agro 2017)
#   X_i = vetor de 25 covariáveis (clima, solo, estrutura agrícola,
#         estrutura econômica, crédito PRONAF, macrorregião)
#
# A Causal Forest constrói árvores honestas (Wager e Athey, 2018) que
# particionam o espaço de X de forma a maximizar a heterogeneidade dos
# efeitos do tratamento, e usa subamostras distintas para escolher os splits
# e para estimar τ(x) dentro de cada folha.
#
# Execução
# -----------------------------------------------------------------------------
#   $ Rscript R/exemplo4.R dados/base_assistencia_tecnica.csv
#
# Saídas (em resultados_modelos/)
# -----------------------------------------------------------------------------
# Resultados em log-pontos (escala nativa do modelo):
#   base_cf_processada.csv         Base efetivamente usada no Causal Forest
#   cf_ate.csv                     ATE (overall, ATT, ATC)
#   cf_calibration.csv             Teste de calibração (Athey-Wager)
#   cf_variable_importance.csv     Importância das variáveis na partição causal
#   cf_blp.csv                     Best Linear Projection do CATE em X
#   cf_cate_por_municipio.csv      CATE estimado, EP e IC95% por município
#   cf_ate_por_subgrupo.csv        ATE por região e por quintis de escala
#   cf_propensity_overlap.csv      Distribuição do propensity score estimado
#   cf_cate_distribuicao.png       Histograma da distribuição dos CATEs
#   cf_cate_por_regiao.png         Boxplot do CATE por macrorregião
#
# Resultados em R$/ha (reexpressão via B·(exp(theta)-1), B = média geom.):
#   cf_ate_reais.csv               ATE/ATT/ATC em R$/ha
#   cf_blp_reais.csv               BLP em R$/ha (linearização)
#   cf_cate_por_municipio_reais.csv  CATE municipal em R$/ha
#   cf_ate_por_subgrupo_reais.csv  Subgrupos em R$/ha
#   cf_cate_por_regiao_reais.png   Boxplot em R$/ha
# =============================================================================


# =============================================================================
# 1. Pacotes
# =============================================================================

required_packages <- c(
  "readr",    # leitura de CSV
  "dplyr",    # manipulação de data frames
  "tidyr",    # drop_na
  "tibble",   # tibbles
  "ggplot2",  # gráficos
  "grf"       # Causal Forest (Wager & Athey)
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
arquivo_base <- ifelse(length(args) >= 1, args[1],
                       "dados/base_assistencia_tecnica.csv")

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
    regiao       = readr::col_character(),
    .default     = readr::col_double()
  )
)

vars_obrigatorias <- c(
  "id_municipio", "regiao",
  "alta_assistencia", "prop_orientacao",
  "produtividade_agropecuaria", "area_media_estab",
  "temperatura_media", "temperatura_maxima", "dias_quentes",
  "precipitacao_anual", "precipitacao_extrema", "dias_secos",
  "carbono_solo", "teor_argila", "teor_areia",
  "pib_total", "pib_agropecuaria", "pib_industria", "pib_servicos",
  "populacao_total", "area_municipio", "pib_percapita",
  "densidade_demografica", "valor_pronaf", "contratos_pronaf",
  "total_estabelecimentos", "area_estabelecimentos_ha"
)

faltantes <- setdiff(vars_obrigatorias, names(base))
if (length(faltantes) > 0) {
  stop("Variáveis ausentes na base: ", paste(faltantes, collapse = ", "))
}

message("Observações lidas: ", nrow(base))
message("Municípios: ",         length(unique(base$id_municipio)))
message("D = 1 (alta assistência): ",
        sum(base$alta_assistencia), " (",
        round(100 * mean(base$alta_assistencia), 1), "%)")


# =============================================================================
# 4. Preparação dos dados
# =============================================================================
# - Filtros mínimos (produtividade observada e positiva).
# - Transformações log nas variáveis de escala.
# - Construção do vetor X numérico (incluindo dummies de macrorregião).
# =============================================================================

log_pos <- function(z) log(pmax(z, 1))

base <- base |>
  dplyr::filter(produtividade_agropecuaria > 0) |>
  dplyr::mutate(
    log_produtividade   = log(produtividade_agropecuaria),
    log_area_media      = log_pos(area_media_estab),
    log_area_estab      = log_pos(area_estabelecimentos_ha),
    log_total_estab     = log_pos(total_estabelecimentos),
    log_pib_total       = log_pos(pib_total),
    log_pib_agropec     = log_pos(pib_agropecuaria),
    log_pib_industria   = log_pos(pib_industria),
    log_pib_servicos    = log_pos(pib_servicos),
    log_populacao       = log_pos(populacao_total),
    log_pib_percapita   = log_pos(pib_percapita),
    log_pronaf          = log1p(valor_pronaf),
    # Dummies de região (Norte = referência)
    regiao_NE  = as.integer(regiao == "Nordeste"),
    regiao_SE  = as.integer(regiao == "Sudeste"),
    regiao_SUL = as.integer(regiao == "Sul"),
    regiao_CO  = as.integer(regiao == "Centro-Oeste")
  )


# =============================================================================
# 5. Especificação Y, W e X
# =============================================================================

vars_clima <- c(
  "temperatura_media", "temperatura_maxima",
  "dias_quentes", "precipitacao_anual",
  "precipitacao_extrema", "dias_secos"
)
vars_solo    <- c("carbono_solo", "teor_argila", "teor_areia")
vars_estrut  <- c("log_area_media", "log_area_estab", "log_total_estab")
vars_socio   <- c(
  "log_pib_total", "log_pib_agropec",
  "log_pib_industria", "log_pib_servicos",
  "log_populacao", "log_pib_percapita",
  "densidade_demografica"
)
vars_credito <- c("log_pronaf", "contratos_pronaf")
vars_regiao  <- c("regiao_NE", "regiao_SE", "regiao_SUL", "regiao_CO")

vars_x <- c(vars_clima, vars_solo, vars_estrut, vars_socio,
            vars_credito, vars_regiao)

dados_cf <- base |>
  dplyr::select(
    id_municipio, regiao,
    Y = log_produtividade,
    W = alta_assistencia,
    dplyr::all_of(vars_x)
  ) |>
  tidyr::drop_na() |>
  as.data.frame()

message("Observações usadas no Causal Forest: ", nrow(dados_cf))
message("Número de covariáveis X: ", length(vars_x),
        " (", length(vars_clima), " clima, ",
        length(vars_solo), " solo, ",
        length(vars_estrut), " estrut. agrícola, ",
        length(vars_socio), " socio, ",
        length(vars_credito), " crédito, ",
        length(vars_regiao), " região)")

readr::write_csv(
  dados_cf,
  file.path(dir_resultados, "base_cf_processada.csv")
)


# =============================================================================
# 6. Ajuste do Causal Forest
# =============================================================================
# Implementação de Wager & Athey (2018): árvores honestas, com cross-fitting
# embutido das regressões nuisance Y.hat = E[Y|X] e W.hat = E[W|X].
# =============================================================================

Y <- dados_cf$Y
W <- dados_cf$W
X <- as.matrix(dados_cf[, vars_x])

message("Estimando Causal Forest (2.000 árvores)...")

cf <- grf::causal_forest(
  X            = X,
  Y            = Y,
  W            = W,
  num.trees    = 2000,
  min.node.size= 5,
  honesty      = TRUE,
  honesty.fraction = 0.5,
  tune.parameters  = "all",
  seed             = 123
)


# =============================================================================
# 7. Diagnóstico de overlap e ATE (overlap-weighted, ATT, ATC)
# =============================================================================
# A estimação do ATE pelo AIPW usa peso 1 / [ê(X)·(1-ê(X))] e explode quando
# algum município tem propensão muito próxima de 0 ou de 1. Em amostras
# brasileiras a exposição à orientação técnica é muito heterogênea entre
# regiões (Sul ~92%, Nordeste ~12%), de forma que essa violação de overlap
# acontece com facilidade. Reportamos então o ATE com **peso de sobreposição**
# (target.sample = "overlap"), que pondera por ê·(1-ê) e dá ênfase à região
# em que tratados e controles coexistem; complementamos com ATT e ATC.
# =============================================================================

# Diagnóstico do propensity score estimado
w_hat <- cf$W.hat
message("Distribuição do propensity score estimado ê(X):")
print(round(stats::quantile(w_hat, c(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)), 3))
message("Município(s) com ê(X) < 0,02: ", sum(w_hat < 0.02))
message("Município(s) com ê(X) > 0,98: ", sum(w_hat > 0.98))

diag_overlap <- tibble::tibble(
  quantil = c("min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max"),
  e_x     = round(c(min(w_hat),
                    stats::quantile(w_hat, c(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)),
                    max(w_hat)), 4)
)
readr::write_csv(
  diag_overlap,
  file.path(dir_resultados, "cf_propensity_overlap.csv")
)

ate_overlap <- grf::average_treatment_effect(cf, target.sample = "overlap")
ate_treated <- grf::average_treatment_effect(cf, target.sample = "treated")
ate_control <- grf::average_treatment_effect(cf, target.sample = "control")

tabela_ate <- tibble::tibble(
  amostra = c("ATE (overlap-weighted)",
              "Apenas tratados (ATT)",
              "Apenas controles (ATC)"),
  theta   = c(ate_overlap["estimate"],
              ate_treated["estimate"],
              ate_control["estimate"]),
  se      = c(ate_overlap["std.err"],
              ate_treated["std.err"],
              ate_control["std.err"])
) |>
  dplyr::mutate(
    ci_inf  = theta - 1.96 * se,
    ci_sup  = theta + 1.96 * se,
    z       = theta / se,
    p_valor = 2 * (1 - pnorm(abs(z)))
  )

print(tabela_ate)

readr::write_csv(
  tabela_ate,
  file.path(dir_resultados, "cf_ate.csv")
)

# Teste de calibração de Athey-Wager: H0 = sem heterogeneidade (slope = 1)
calib <- grf::test_calibration(cf)
calib_df <- tibble::tibble(
  termo        = rownames(calib),
  estimativa   = calib[, "Estimate"],
  erro_padrao  = calib[, "Std. Error"],
  z            = calib[, "t value"],
  p_valor      = calib[, "Pr(>t)"]
)
print(calib_df)
readr::write_csv(
  calib_df,
  file.path(dir_resultados, "cf_calibration.csv")
)


# =============================================================================
# 8. Importância das variáveis na partição causal
# =============================================================================

vi <- grf::variable_importance(cf)
importancia <- tibble::tibble(
  variavel    = vars_x,
  importancia = as.numeric(vi)
) |>
  dplyr::arrange(dplyr::desc(importancia))

print(head(importancia, 10))

readr::write_csv(
  importancia,
  file.path(dir_resultados, "cf_variable_importance.csv")
)


# =============================================================================
# 9. Best Linear Projection (BLP) do CATE
# =============================================================================
# Projeta τ̂(X) em um conjunto reduzido de variáveis para interpretar
# direções da heterogeneidade.
# =============================================================================

vars_blp <- c("log_area_media", "log_pib_agropec",
              "temperatura_media", "precipitacao_anual",
              "regiao_NE", "regiao_SE", "regiao_SUL", "regiao_CO")

blp <- grf::best_linear_projection(cf, X[, vars_blp])
blp_df <- tibble::tibble(
  variavel    = rownames(blp),
  coeficiente = blp[, "Estimate"],
  erro_padrao = blp[, "Std. Error"],
  z           = blp[, "t value"],
  p_valor     = blp[, "Pr(>|t|)"]
)
print(blp_df)
readr::write_csv(
  blp_df,
  file.path(dir_resultados, "cf_blp.csv")
)


# =============================================================================
# 10. CATE por município (com variância) e por subgrupos
# =============================================================================

tau_hat <- predict(cf, estimate.variance = TRUE)
cate_munic <- tibble::tibble(
  id_municipio = dados_cf$id_municipio,
  regiao       = dados_cf$regiao,
  W            = dados_cf$W,
  tau          = tau_hat$predictions,
  variancia    = tau_hat$variance.estimates,
  se           = sqrt(tau_hat$variance.estimates)
) |>
  dplyr::mutate(
    ci_inf = tau - 1.96 * se,
    ci_sup = tau + 1.96 * se
  )

readr::write_csv(
  cate_munic,
  file.path(dir_resultados, "cf_cate_por_municipio.csv")
)

# ATE por região
ate_por_regiao <- cate_munic |>
  dplyr::group_by(regiao) |>
  dplyr::summarise(
    n           = dplyr::n(),
    tau_medio   = mean(tau),
    tau_p25     = stats::quantile(tau, 0.25),
    tau_mediano = stats::median(tau),
    tau_p75     = stats::quantile(tau, 0.75),
    .groups = "drop"
  )

# ATE por quintis de escala (área média do estabelecimento)
dados_cf$quintil_escala <- dplyr::ntile(dados_cf$log_area_media, 5)
ate_por_escala <- cate_munic |>
  dplyr::mutate(quintil_escala = dados_cf$quintil_escala) |>
  dplyr::group_by(quintil_escala) |>
  dplyr::summarise(
    n           = dplyr::n(),
    tau_medio   = mean(tau),
    tau_p25     = stats::quantile(tau, 0.25),
    tau_mediano = stats::median(tau),
    tau_p75     = stats::quantile(tau, 0.75),
    .groups = "drop"
  )

subgrupos <- dplyr::bind_rows(
  ate_por_regiao |>
    dplyr::mutate(grupo = "Macrorregião") |>
    dplyr::rename(estrato = regiao),
  ate_por_escala |>
    dplyr::mutate(grupo = "Quintil de escala",
                  estrato = as.character(quintil_escala)) |>
    dplyr::select(-quintil_escala)
) |>
  dplyr::select(grupo, estrato, n, tau_medio, tau_p25, tau_mediano, tau_p75)

print(subgrupos)
readr::write_csv(
  subgrupos,
  file.path(dir_resultados, "cf_ate_por_subgrupo.csv")
)


# =============================================================================
# 11. Reexpressão em R$/ha (base = média geométrica amostral)
# =============================================================================
# Os efeitos do Causal Forest são estimados em log-pontos porque
# Y = log(produtividade_agropecuaria). Para a leitura econômica
# reexpressamos esses efeitos em R$/ha pela transformação multiplicativa
#
#     efeito_R$_ha = B * (exp(theta) - 1),
#
# em que B é a produtividade de referência. Adotamos como B a média
# geométrica amostral da produtividade, B = exp(mean(Y)) — invariante a
# mudanças de unidade e equivalente à mediana da produtividade quando a
# distribuição é log-normal.
#
# Erros-padrão em R$/ha vêm do método delta:
#     SE_R$_ha = B * exp(theta) * SE_theta.
# Os limites do IC 95% são transformados diretamente pela monotonicidade
# da exponencial. Os valores de z e p-valor permanecem inalterados — a
# hipótese nula theta = 0 equivale a efeito = 0.
#
# OBS sobre unidades: na base original, produtividade_agropecuaria é
# valor_producao (em R$ 1.000) dividido por area_estabelecimentos (ha),
# portanto está em R$ mil/ha. Multiplicamos B por 1.000 para reportar
# o efeito em R$/ha.
# =============================================================================

B_mil_ha   <- exp(mean(dados_cf$Y))
B_reais_ha <- B_mil_ha * 1000  # R$ por hectare

message("Base de referencia (media geometrica amostral):")
message(sprintf("  B = R$ %.0f/ha", B_reais_ha))

# --- Tabela 5.2 em R$/ha ---------------------------------------------------
tabela_ate_reais <- tabela_ate |>
  dplyr::mutate(
    theta_reais  = B_reais_ha * (exp(theta) - 1),
    se_reais     = B_reais_ha * exp(theta) * se,
    ci_inf_reais = B_reais_ha * (exp(ci_inf) - 1),
    ci_sup_reais = B_reais_ha * (exp(ci_sup) - 1)
  ) |>
  dplyr::select(amostra,
                theta_reais, se_reais,
                ci_inf_reais, ci_sup_reais,
                z, p_valor)

print(tabela_ate_reais)
readr::write_csv(
  tabela_ate_reais,
  file.path(dir_resultados, "cf_ate_reais.csv")
)

# --- CATE municipal em R$/ha -----------------------------------------------
cate_munic_reais <- cate_munic |>
  dplyr::mutate(
    tau_reais    = B_reais_ha * (exp(tau) - 1),
    se_reais     = B_reais_ha * exp(tau) * se,
    ci_inf_reais = B_reais_ha * (exp(ci_inf) - 1),
    ci_sup_reais = B_reais_ha * (exp(ci_sup) - 1)
  ) |>
  dplyr::select(id_municipio, regiao, W,
                tau_reais, se_reais, ci_inf_reais, ci_sup_reais)

readr::write_csv(
  cate_munic_reais,
  file.path(dir_resultados, "cf_cate_por_municipio_reais.csv")
)

# --- BLP em R$/ha (linearizacao: marginal = B * coef em torno de theta = 0)
blp_reais <- blp_df |>
  dplyr::mutate(
    coef_reais        = B_reais_ha * coeficiente,
    erro_padrao_reais = B_reais_ha * erro_padrao
  ) |>
  dplyr::select(variavel,
                coef_reais, erro_padrao_reais,
                z, p_valor)

print(blp_reais)
readr::write_csv(
  blp_reais,
  file.path(dir_resultados, "cf_blp_reais.csv")
)

# --- Subgrupos (medias do CATE) em R$/ha -----------------------------------
subgrupos_reais <- subgrupos |>
  dplyr::mutate(
    tau_medio_reais   = B_reais_ha * (exp(tau_medio)   - 1),
    tau_p25_reais     = B_reais_ha * (exp(tau_p25)     - 1),
    tau_mediano_reais = B_reais_ha * (exp(tau_mediano) - 1),
    tau_p75_reais     = B_reais_ha * (exp(tau_p75)     - 1)
  ) |>
  dplyr::select(grupo, estrato, n,
                tau_medio_reais, tau_p25_reais, tau_mediano_reais, tau_p75_reais)

print(subgrupos_reais)
readr::write_csv(
  subgrupos_reais,
  file.path(dir_resultados, "cf_ate_por_subgrupo_reais.csv")
)


# =============================================================================
# 12. Graficos: distribuicao do CATE e CATE por regiao
# =============================================================================

g_hist <- ggplot2::ggplot(
    cate_munic, ggplot2::aes(x = tau)
  ) +
  ggplot2::geom_histogram(bins = 40, fill = "steelblue", color = "white") +
  ggplot2::geom_vline(xintercept = 0, linetype = 2) +
  ggplot2::geom_vline(xintercept = ate_overlap["estimate"], color = "red") +
  ggplot2::labs(
    title    = "Distribuicao dos efeitos heterogeneos do tratamento (tau)",
    subtitle = paste0("Linha vermelha: ATE (overlap-weighted) = ",
                      round(ate_overlap["estimate"], 3)),
    x = "tau(X) - variacao esperada em log(produtividade)",
    y = "Numero de municipios"
  ) +
  ggplot2::theme_minimal(base_size = 12)

ggplot2::ggsave(
  filename = file.path(dir_resultados, "cf_cate_distribuicao.png"),
  plot     = g_hist,
  width    = 9, height = 5, dpi = 300
)

g_box <- ggplot2::ggplot(
    cate_munic,
    ggplot2::aes(x = regiao, y = tau, fill = regiao)
  ) +
  ggplot2::geom_boxplot(alpha = 0.7, outlier.size = 0.7) +
  ggplot2::geom_hline(yintercept = 0, linetype = 2) +
  ggplot2::labs(
    title = "CATE estimado por macrorregiao",
    x     = "Macrorregiao",
    y     = "tau(X)"
  ) +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::theme(legend.position = "none")

ggplot2::ggsave(
  filename = file.path(dir_resultados, "cf_cate_por_regiao.png"),
  plot     = g_box,
  width    = 9, height = 5, dpi = 300
)

# Versao paralela em R$/ha — mesma estrutura, eixo y reescalado
g_box_reais <- ggplot2::ggplot(
    cate_munic_reais,
    ggplot2::aes(x = regiao, y = tau_reais, fill = regiao)
  ) +
  ggplot2::geom_boxplot(alpha = 0.7, outlier.size = 0.7) +
  ggplot2::geom_hline(yintercept = 0, linetype = 2) +
  ggplot2::labs(
    title = "CATE estimado por macrorregiao (R$/ha)",
    x     = "Macrorregiao",
    y     = "tau(X) em R$/ha"
  ) +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::theme(legend.position = "none")

ggplot2::ggsave(
  filename = file.path(dir_resultados, "cf_cate_por_regiao_reais.png"),
  plot     = g_box_reais,
  width    = 9, height = 5, dpi = 300
)


# =============================================================================
# 13. Fim
# =============================================================================

message("Resultados salvos em: ", dir_resultados)
s.png"),
  plot     = g_box_reais,
  width    = 9, height = 5, dpi = 300
)


# =============================================================================
# 13. Fim
# =============================================================================

message("Resultados salvos em: ", dir_resultados)
