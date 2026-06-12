#!/usr/bin/env Rscript

# =============================================================================
# Script:        exemplo2.R
# Capítulo:      Machine Learning em Avaliação de Impacto na Economia
# Exemplo:       2 - Classificação de (in)segurança alimentar em domicílios
#                    brasileiros, POF 2017-2018, com a EBIA como variável
#                    dependente observada
# Autor:         (preencher)
# Última revisão: 2025
#
# Objetivo
# -----------------------------------------------------------------------------
# Comparar dois modelos de classificação multinomial para a variável
# 'classe_ebia' (Escala Brasileira de Insegurança Alimentar) em domicílios
# da POF 2017-2018:
#
#   (1) Regressão logística multinomial (nnet::multinom)
#   (2) Random Forest multiclasses     (ranger::ranger)
#
# A EBIA tem 4 classes:
#   - Segurança alimentar
#   - Insegurança alimentar leve
#   - Insegurança alimentar moderada
#   - Insegurança alimentar grave
#
# Estratégia metodológica
# -----------------------------------------------------------------------------
#   - Unidade de análise: domicílio.
#   - Excluem-se identificadores e variáveis do plano amostral (uf,
#     estrato_pof, cod_upa, num_dom).
#   - Renda per capita é expressa em R$ mil para facilitar a
#     interpretação dos odds ratios.
#   - Valores ausentes em numéricas são imputados pela mediana.
#   - Valores ausentes em fatores recebem o nível "Ignorado".
#   - Split estratificado 80/20 (treino/teste) via caret::createDataPartition.
#   - Importância das variáveis no Random Forest: importância por
#     permutação (importance = "permutation" em ranger::ranger()).
#   - Reprodutibilidade: set.seed(123) é executado uma vez no início do
#     script e novamente imediatamente antes da partição treino/teste. O
#     Random Forest recebe seed=123 explicitamente em ranger().
#
# Execução
# -----------------------------------------------------------------------------
#   $ Rscript R/exemplo2.R dados/base_ebia.csv
#
# Se nenhum argumento for fornecido, o script tenta ler "dados/base_ebia.csv"
# a partir do diretório de trabalho atual.
#
# Saídas (em resultados_modelos/)
# -----------------------------------------------------------------------------
#   distribuicao_classes.csv         Distribuição das 4 classes da EBIA
#   metricas_classificacao.csv       Accuracy, F1 macro, Bal. accuracy e Kappa
#   matriz_confusao_logit.csv        Matriz de confusão da logística
#   matriz_confusao_rf.csv           Matriz de confusão do Random Forest
#   importancia_variaveis_rf.csv     Importância das variáveis (RF)
#   importancia_variaveis_rf.png     Gráfico das 15 variáveis mais importantes
#   odds_ratios_logit.csv            Odds ratios e IC95% (top 5 vars do RF)
#   odds_ratios_logit.png            Forest plot dos odds ratios
#   previsoes_teste.csv              Previsões dos dois modelos no teste
# =============================================================================


# =============================================================================
# 1. Pacotes
# =============================================================================

required_packages <- c(
  "readr",     # leitura de CSV
  "dplyr",     # manipulação de data frames
  "janitor",   # clean_names()
  "forcats",   # manipulação de fatores
  "caret",     # createDataPartition (split estratificado)
  "nnet",      # multinom (regressão logística multinomial)
  "ranger",    # Random Forest multiclasses
  "yardstick", # métricas de classificação
  "ggplot2",   # gráficos
  "tibble"     # tibbles
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
arquivo_base <- ifelse(length(args) >= 1, args[1], "dados/base_ebia.csv")

dir_resultados <- "resultados_modelos"
dir.create(dir_resultados, recursive = TRUE, showWarnings = FALSE)

message("Lendo base: ", arquivo_base)

if (!file.exists(arquivo_base)) {
  stop("Arquivo de dados não encontrado: ", arquivo_base)
}


# =============================================================================
# 3. Leitura e validação da base
# =============================================================================

base <- readr::read_csv(arquivo_base, show_col_types = FALSE) |>
  janitor::clean_names()

if (!("classe_ebia" %in% names(base))) {
  stop("A base precisa conter a variável dependente 'classe_ebia'.")
}


# =============================================================================
# 4. Preparação dos dados
# =============================================================================
# - classe_ebia como fator com níveis em ordem crescente de gravidade.
# - Remoção de identificadores e variáveis do plano amostral.
# - Remoção de colunas totalmente NA.
# - Conversão da renda per capita para R$ mil (facilita interpretação dos OR).
# - Remoção de variáveis constantes.
# =============================================================================

base <- base |>
  dplyr::mutate(
    classe_ebia = factor(
      classe_ebia,
      levels = c(
        "Segurança alimentar",
        "Insegurança alimentar leve",
        "Insegurança alimentar moderada",
        "Insegurança alimentar grave"
      )
    )
  ) |>
  dplyr::filter(!is.na(classe_ebia))

# Identificadores e variáveis do plano amostral da POF
vars_excluir <- c("uf", "estrato_pof", "cod_upa", "num_dom")

base_modelo <- base |>
  dplyr::select(-dplyr::any_of(vars_excluir)) |>
  dplyr::select(where(~ !all(is.na(.x))))

# Renda em R$ mil
base_modelo <- base_modelo |>
  dplyr::mutate(
    percapita_mil = renda_percapita / 1000
  ) |>
  dplyr::select(-renda_percapita)

# Remove variáveis constantes (exceto a variável dependente)
vars_constantes <- names(base_modelo)[
  sapply(base_modelo, function(x) length(unique(x[!is.na(x)])) <= 1)
]
vars_constantes <- setdiff(vars_constantes, "classe_ebia")

if (length(vars_constantes) > 0) {
  message(
    "Removendo variáveis constantes: ",
    paste(vars_constantes, collapse = ", ")
  )
  base_modelo <- base_modelo |>
    dplyr::select(-dplyr::all_of(vars_constantes))
}


# =============================================================================
# 5. Tratamento de valores ausentes
# =============================================================================
# - Numéricas: imputação pela mediana.
# - Fatores (exceto a variável dependente): NA recebe o nível "Ignorado".
#   Implementação portátil que funciona em qualquer versão de forcats.
# =============================================================================

ignorar_na_fator <- function(x) {
  if (is.factor(x)) {
    factor(ifelse(is.na(x), "Ignorado", as.character(x)))
  } else {
    x
  }
}

base_modelo <- base_modelo |>
  dplyr::mutate(
    dplyr::across(
      where(is.numeric),
      ~ ifelse(
        is.na(.x) | is.nan(.x) | !is.finite(.x),
        median(.x, na.rm = TRUE),
        .x
      )
    )
  ) |>
  dplyr::mutate(
    dplyr::across(-classe_ebia, ignorar_na_fator)
  )


# =============================================================================
# 6. Distribuição das classes (descrição)
# =============================================================================

dist_classes <- base_modelo |>
  dplyr::count(classe_ebia) |>
  dplyr::mutate(prop = n / sum(n))

print(dist_classes)

readr::write_csv(
  dist_classes,
  file.path(dir_resultados, "distribuicao_classes.csv")
)


# =============================================================================
# 7. Split treino/teste (80/20 estratificado por classe_ebia)
# =============================================================================

set.seed(123)
idx <- caret::createDataPartition(
  y     = base_modelo$classe_ebia,
  p     = 0.8,
  list  = FALSE
)

train <- base_modelo[idx,  ]
test  <- base_modelo[-idx, ]

# Garante que fatores no teste têm os mesmos níveis do treino
for (v in names(train)) {
  if (is.factor(train[[v]]) && v %in% names(test)) {
    test[[v]] <- factor(test[[v]], levels = levels(train[[v]]))
  }
}

message("Treino: ", nrow(train), " obs / Teste: ", nrow(test), " obs")


# =============================================================================
# 8. Funções auxiliares
# =============================================================================

# Linha de métricas para um modelo de classificação multinomial
avaliar_modelo <- function(truth, estimate, nome_modelo) {
  tibble::tibble(
    modelo       = nome_modelo,
    accuracy     = yardstick::accuracy_vec(truth = truth, estimate = estimate),
    macro_f1     = yardstick::f_meas_vec(
      truth     = truth,
      estimate  = estimate,
      estimator = "macro"
    ),
    bal_accuracy = yardstick::bal_accuracy_vec(
      truth     = truth,
      estimate  = estimate,
      estimator = "macro"
    ),
    kappa        = yardstick::kap_vec(truth = truth, estimate = estimate)
  )
}


# =============================================================================
# 9. Modelo 1 — Regressão logística multinomial
# =============================================================================

message("Estimando regressão logística multinomial...")

modelo_logit <- nnet::multinom(
  classe_ebia ~ .,
  data    = train,
  trace   = FALSE,
  MaxNWts = 50000,
  maxit   = 500
)

pred_logit <- predict(modelo_logit, newdata = test, type = "class")
pred_logit <- factor(pred_logit, levels = levels(test$classe_ebia))


# =============================================================================
# 10. Modelo 2 — Random Forest multiclasses
# =============================================================================

message("Estimando Random Forest multiclasses...")

p <- ncol(train) - 1  # número de preditores

modelo_rf <- ranger::ranger(
  classe_ebia ~ .,
  data          = train,
  num.trees     = 500,
  mtry          = max(1, floor(sqrt(p))),
  min.node.size = 10,
  probability   = FALSE,
  importance    = "permutation",
  seed          = 123
)

pred_rf <- predict(modelo_rf, data = test)$predictions
pred_rf <- factor(pred_rf, levels = levels(test$classe_ebia))


# =============================================================================
# 11. Métricas comparativas no conjunto de teste
# =============================================================================

metricas <- dplyr::bind_rows(
  avaliar_modelo(test$classe_ebia, pred_logit, "Regressão logística multinomial"),
  avaliar_modelo(test$classe_ebia, pred_rf,    "Random Forest")
)

print(metricas)

readr::write_csv(
  metricas,
  file.path(dir_resultados, "metricas_classificacao.csv")
)


# =============================================================================
# 12. Matrizes de confusão
# =============================================================================

conf_logit <- tibble::tibble(
  verdade  = test$classe_ebia,
  previsto = pred_logit
) |>
  dplyr::count(verdade, previsto)

conf_rf <- tibble::tibble(
  verdade  = test$classe_ebia,
  previsto = pred_rf
) |>
  dplyr::count(verdade, previsto)

readr::write_csv(
  conf_logit,
  file.path(dir_resultados, "matriz_confusao_logit.csv")
)

readr::write_csv(
  conf_rf,
  file.path(dir_resultados, "matriz_confusao_rf.csv")
)

# Imprime as matrizes em formato yardstick
print(
  yardstick::conf_mat(
    tibble::tibble(truth = test$classe_ebia, estimate = pred_logit),
    truth    = truth,
    estimate = estimate
  )
)

print(
  yardstick::conf_mat(
    tibble::tibble(truth = test$classe_ebia, estimate = pred_rf),
    truth    = truth,
    estimate = estimate
  )
)


# =============================================================================
# 13. Importância de variáveis (Random Forest, por permutação)
# =============================================================================

importancia_rf <- tibble::tibble(
  variavel    = names(modelo_rf$variable.importance),
  importancia = as.numeric(modelo_rf$variable.importance)
) |>
  dplyr::arrange(dplyr::desc(importancia))

readr::write_csv(
  importancia_rf,
  file.path(dir_resultados, "importancia_variaveis_rf.csv")
)

grafico_importancia <- importancia_rf |>
  dplyr::slice_head(n = 15) |>
  ggplot2::ggplot(
    ggplot2::aes(x = reorder(variavel, importancia), y = importancia)
  ) +
  ggplot2::geom_col() +
  ggplot2::coord_flip() +
  ggplot2::labs(
    title = "Random Forest: importância das variáveis (permutação)",
    x     = "",
    y     = "Escore de importância (permutação)"
  ) +
  ggplot2::theme_minimal(base_size = 12)

ggplot2::ggsave(
  filename = file.path(dir_resultados, "importancia_variaveis_rf.png"),
  plot     = grafico_importancia,
  width    = 8,
  height   = 5,
  dpi      = 300
)


# =============================================================================
# 14. Odds ratios da regressão logística (5 vars mais importantes do RF)
# =============================================================================
# Extrai coeficientes e erros-padrão de nnet::multinom, calcula
# OR = exp(beta) e IC95% = exp(beta +/- 1.96 * se).
# =============================================================================

top5_vars <- importancia_rf |>
  dplyr::slice_head(n = 5) |>
  dplyr::pull(variavel)

coef_mat <- nnet:::coef.multinom(modelo_logit)
sum_logit <- nnet:::summary.multinom(modelo_logit)
se_mat   <- sum_logit$standard.errors

coef_logit <- as.data.frame(as.table(coef_mat)) |>
  dplyr::rename(y.level = Var1, term = Var2, beta = Freq) |>
  dplyr::left_join(
    as.data.frame(as.table(se_mat)) |>
      dplyr::rename(y.level = Var1, term = Var2, se = Freq),
    by = c("y.level", "term")
  ) |>
  dplyr::mutate(
    estimate  = exp(beta),
    conf.low  = exp(beta - 1.96 * se),
    conf.high = exp(beta + 1.96 * se)
  )

odds_ratios <- coef_logit |>
  dplyr::filter(
    term %in% top5_vars,
    term != "(Intercept)"
  ) |>
  dplyr::mutate(
    term    = factor(term,    levels = top5_vars),
    y.level = factor(
      y.level,
      levels = c(
        "Insegurança alimentar leve",
        "Insegurança alimentar moderada",
        "Insegurança alimentar grave"
      )
    )
  )

readr::write_csv(
  odds_ratios,
  file.path(dir_resultados, "odds_ratios_logit.csv")
)

# Mapeamento de nomes de variáveis para rótulos legíveis no gráfico
labels_termos <- c(
  percapita_mil = "Renda per capita (R$ mil)",
  num_comodos   = "Número de cômodos",
  num_banheiros = "Número de banheiros",
  num_moradores = "Número de moradores",
  fontes_renda  = "Número de fontes de renda"
)

grafico_or <- ggplot2::ggplot(
    odds_ratios,
    ggplot2::aes(
      x     = estimate,
      y     = y.level,
      xmin  = conf.low,
      xmax  = conf.high,
      color = term
    )
  ) +
  ggplot2::geom_vline(xintercept = 1, linetype = 2, linewidth = 0.5) +
  ggplot2::geom_point(size = 2) +
  ggplot2::geom_errorbar(width = 0.15, orientation = "y") +
  ggplot2::facet_wrap(
    ~ term,
    scales   = "fixed",
    nrow     = 1,
    labeller = ggplot2::labeller(term = labels_termos)
  ) +
  ggplot2::scale_y_discrete(
    labels = c(
      "Insegurança alimentar leve"     = "Leve",
      "Insegurança alimentar moderada" = "Moderada",
      "Insegurança alimentar grave"    = "Grave"
    )
  ) +
  ggplot2::labs(
    x = "Odds ratio da regressão logística multinomial (IC95%)",
    y = "Insegurança alimentar no domicílio"
  ) +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::theme(
    legend.position  = "none",
    strip.text       = ggplot2::element_text(face = "bold"),
    panel.background = ggplot2::element_rect(fill = "gray95", color = NA),
    panel.grid.major = ggplot2::element_line(color = "white"),
    panel.grid.minor = ggplot2::element_blank()
  )

ggplot2::ggsave(
  filename = file.path(dir_resultados, "odds_ratios_logit.png"),
  plot     = grafico_or,
  width    = 11,
  height   = 5,
  dpi      = 300
)


# =============================================================================
# 15. Previsões no conjunto de teste
# =============================================================================

previsoes <- tibble::tibble(
  classe_observada = test$classe_ebia,
  pred_logit       = pred_logit,
  pred_rf          = pred_rf
)

readr::write_csv(
  previsoes,
  file.path(dir_resultados, "previsoes_teste.csv")
)


# =============================================================================
# 16. Fim
# =============================================================================

message("Resultados salvos em: ", dir_resultados)
