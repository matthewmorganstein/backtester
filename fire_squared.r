library(shiny)
library(shinyheatmap)
library(dplyr)
library(tidyr)
library(lubridate)

# Define TradingStrategy class (from your original code)
TradingStrategy <- R6::R6Class(
  "TradingStrategy",
  public = list(
    square_threshold = NULL,
    initialize = function(square_threshold) {
      self$square_threshold <- square_threshold
    },
    pointland_signal = function(df) {
      prev_low <- df$low %>% dplyr::lag(1)
      prev_high <- df$high %>% dplyr::lag(1)
      buy_signal <- (df$close < prev_low) & 
        ((df$r_1 > self$square_threshold) | (df$r_2 > self$square_threshold))
      sell_signal <- (df$close > prev_high) & 
        ((df$r_1 > self$square_threshold) | (df$r_2 > self$square_threshold))
      signals <- rep(0, nrow(df))
      signals[buy_signal] <- 1
      signals[sell_signal] <- -1
      return(signals)
    }
  )
)

# Sample data preparation (replace with your actual data)
market_data <- data.frame(
  asset = rep(c("AAPL", "MSFT"), each = 5),
  timestamp = rep(seq(as.POSIXct("2025-05-01 09:00:00"), by = "hour", length.out = 5), 2),
  close = c(150, 151, 149, 152, 150, 300, 305, 298, 302, 301),
  high = c(152, 153, 150, 153, 151, 305, 310, 300, 305, 303),
  low = c(149, 150, 148, 150, 149, 298, 300, 295, 298, 299),
  r_1 = c(0.8, 1.2, 1.1, 0.9, 1.3, 0.7, 1.0, 1.4, 0.8, 1.2),
  r_2 = c(0.9, 1.1, 1.0, 0.8, 1.2, 0.6, 0.9, 1.3, 0.7, 1.1)
)

strategy <- TradingStrategy$new(square_threshold = 1.0)

timeframes <- c("1 hour", "4 hours", "1 day")
signal_data <- lapply(timeframes, function(tf) {
  tf_data <- market_data %>%
    group_by(asset, timestamp = floor_date(timestamp, tf)) %>%
    summarise(
      close = last(close),
      high = max(high),
      low = min(low),
      r_1 = mean(r_1),
      r_2 = mean(r_2),
      .groups = "drop"
    )
  tf_data <- tf_data %>%
    group_by(asset) %>%
    mutate(signal = strategy$pointland_signal(.)) %>%
    ungroup()
  tf_data$timeframe <- tf
  return(tf_data)
}) %>% bind_rows()

heatmap_data <- signal_data %>%
  pivot_wider(
    names_from = timestamp,
    values_from = signal,
    id_cols = c(asset, timeframe),
    values_fill = 0
  )

# Shiny UI with pixelated, fiery aesthetic
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
      body {
        background-color: #000;
        color: #fff;
        font-family: 'Press Start 2P', cursive;
        overflow: hidden;
      }
      .pixel-title {
        color: #ff4500;
        font-size: 20px;
        text-align: center;
        margin: 20px;
        text-shadow: 2px 2px #ffd700, 4px 4px #ff0000;
      }
      .sidebar {
        background: linear-gradient(180deg, #ff4500, #ffd700);
        border: 4px solid #00f;
        border-right: 4px solid #f00;
        padding: 15px;
        border-radius: 0;
      }
      .lego-button {
        background-color: #00ffff;
        color: #000;
        border: 3px solid #f00;
        border-right: 3px solid #00f;
        border-bottom: 3px solid #00f;
        padding: 10px;
        font-family: 'Press Start 2P', cursive;
        font-size: 12px;
        border-radius: 0;
        box-shadow: 2px 2px #000;
      }
      .lego-button:hover {
        background-color: #00cccc;
      }
      .heatmap-container {
        background-color: #000;
        border: 4px solid #f00;
        border-right: 4px solid #00f;
        border-bottom: 4px solid #00f;
        padding: 10px;
      }
      .select-input {
        background-color: #ffd700;
        color: #000;
        border: 3px solid #f00;
        border-right: veterinarian 3px solid #00f;
        border-bottom: 3px solid #00f;
        font-family: 'Press Start 2P', cursive;
        font-size: 12px;
      }
      .star {
        position: absolute;
        color: #ffd700;
        font-size: 12px;
      }
      .cross {
        position: absolute;
        color: #ff0000;
        font-size: 12px;
      }
      .square {
        position: absolute;
        color: #00ffff;
        font-size: 12px;
      }
    "))
  ),
  # Add pixelated stars, crosses, and squares in the background
  lapply(1:20, function(i) {
    tags$div(
      class = sample(c("star", "cross", "square"), 1),
      style = paste0(
        "left:", runif(1, 0, 100), "%;",
        "top:", runif(1, 0, 100), "%;"
      ),
      sample(c("✦", "✖", "■"), 1)
    )
  }),
  div(class = "pixel-title", "🔥 Pixel Trading Heatmap 🔥"),
  sidebarLayout(
    sidebarPanel(
      class = "sidebar",
      selectInput("timeframe", "Timeframe", choices = timeframes, class = "select-input"),
      actionButton("update", "Update", class = "lego-button")
    ),
    mainPanel(
      div(class = "heatmap-container", uiOutput("heatmap"))
    )
  )
)

# Shiny server
server <- function(input, output, session) {
  observeEvent(input$update, {
    plot_data <- heatmap_data %>%
      filter(timeframe == input$timeframe) %>%
      select(-timeframe) %>%
      column_to_rownames("asset")
    
    output$heatmap <- renderUI({
      mat <- as.matrix(plot_data)
      shinyheatmap::renderHeatmap(
        mat,
        lowColor = "#ff0000",    # Sell (-1)
        midColor = "#000000",    # Hold (0)
        highColor = "#00ff00",   # Buy (1)
        title = paste("Signals for", input$timeframe),
        titleFont = "'Press Start 2P', cursive",
        titleColor = "#ffd700",
        borderColor = "#00ffff"
      )
    })
  })
}

# Run the app
shinyApp(ui, server)
