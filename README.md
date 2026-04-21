# Binance Trend Trading Bot

Bot Python autonome pour `BTCUSDT` et `ETHUSDT` avec :

- recuperation des chandeliers Binance Spot
- prediction de tendance par score d'indicateurs
- backtest avec frais, stop-loss et take-profit
- mode signal
- mode execution Binance optionnel (`test-order` ou reel) pour achat et vente

## Strategie

Le bot travaille par defaut en journalier (`1d`) et considere une tendance haussiere quand plusieurs conditions sont alignees :

- `close > SMA 200`
- `EMA 20 > EMA 50`
- histogramme MACD positif
- `RSI` entre 52 et 72
- cassure proche du plus haut des 20 dernieres bougies

Le bot entre en position si le score haussier est suffisant. Il sort si la structure devient baissiere, ou si un stop-loss / take-profit est touche.

Presets disponibles :

- `trend_1d` : preset journalier tres selectif, retune ici pour mieux filtrer `ETHUSDT` sur 2026
- `intraday_1h` : preset intraday momentum, retune ici pour `ETHUSDT` sur 2026
- `hybrid_1h` : preset 1h a regimes `bull / bear / range`, avec logique d'entree/sortie differente selon le contexte
- `hybrid_ml_1h` : preset hybride 1h avec filtre machine learning qui decide si le contexte vaut un trade ou non

## Prerequis

- Python 3.11+
- `requests`

## Docker

`docker-compose.yml` est utile ici parce que le bot est un process long-running avec etat local persistant :

- `.runtime/` : portefeuille papier, etat de boucle, historique
- `.cache/market_data/` : cache des chandeliers

Preparation :

```bash
cp .env.example .env
```

Build :

```bash
docker compose build
```

Lancer le bot papier en continu :

```bash
docker compose up -d paper-bot
```

Suivre les logs :

```bash
docker compose logs -f paper-bot
```

Arreter :

```bash
docker compose down
```

Lancer un backtest ponctuel dans Docker :

```bash
docker compose run --rm backtest
```

## Exemples

Precharger le cache local des chandeliers :

```bash
python3 cache_market_data.py --symbols ETHUSDT BTCUSDT --intervals 1h 1d --start 2025-01-01 --end 2026-04-20
```

Le client recharge automatiquement le cache local dans `.cache/market_data/` avant d'interroger Binance.
Si `pyarrow` est disponible, le format utilise sera `Feather`, sinon un cache `JSON` local est utilise sans installation supplementaire.

Backtest sur les 2 actifs :

```bash
python3 bot.py backtest --symbols BTCUSDT ETHUSDT --start 2024-01-01 --end 2026-04-20
```

Backtest intraday 1h :

```bash
python3 bot.py backtest --symbols BTCUSDT ETHUSDT --preset intraday_1h --start 2024-01-01 --end 2026-04-20
```

Dernier signal :

```bash
python3 bot.py signal --symbol BTCUSDT
```

Dernier signal intraday 1h :

```bash
python3 bot.py signal --symbol BTCUSDT --preset intraday_1h
```

Dernier signal hybride 1h :

```bash
python3 bot.py signal --symbol ETHUSDT --preset hybrid_1h
```

Dernier signal hybride 1h + ML :

```bash
python3 bot.py signal --symbol ETHUSDT --preset hybrid_ml_1h
```

Execution de test Binance sans ordre reel :

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
python3 bot.py live --symbol BTCUSDT --quote-order-qty 250 --test-order
```

Execution reelle :

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
python3 bot.py live --symbol BTCUSDT --quote-order-qty 250
```

## Remarques

- Le bot n'offre aucune garantie de profit.
- Les resultats de backtest ne garantissent pas les performances futures.
- Le mode live n'est pas lance automatiquement.
