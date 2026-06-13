cat > /tmp/brain.py << 'ENDOFFILE'
import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import os
import json
import datetime
import requests

# ============================================================
#  CONFIG GLOBALE
# ============================================================
MODEL_PATH   = "trading_model.pth"
MEMORY_PATH  = "mozai_memory.json"
LOG_PATH     = "trade_history.json"
TELEGRAM_TOKEN = "TON_TOKEN_TELEGRAM"
TELEGRAM_CHAT  = "TON_CHAT_ID"

SEQ_LEN      = 60   # fenêtre temporelle
INPUT_SIZE   = 40   # nombre d'indicateurs
HIDDEN_SIZE  = 512  # neurones LSTM
NUM_LAYERS   = 6    # couches LSTM
DENSE_SIZE   = 256  # neurones dense

# ============================================================
#  ARCHITECTURE NEURONALE — 10 000+ paramètres actifs
# ============================================================
class AttentionLayer(nn.Module):
    """Mécanisme d'attention : le réseau décide QUOI regarder."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn    = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        attn_weights = torch.tanh(self.attn(lstm_out))
        attn_weights = torch.softmax(self.context(attn_weights), dim=1)
        return (attn_weights * lstm_out).sum(dim=1)


class MarketSentimentHead(nn.Module):
    """Tête spécialisée : détecte les anomalies et le 'pressentiment'."""
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 4)  # [normal, anomalie, manipulation, incertitude]
        )

    def forward(self, x):
        return self.net(x)


class TradingLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodeur LSTM bidirectionnel
        self.lstm = nn.LSTM(
            INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Couche d'attention
        self.attention = AttentionLayer(HIDDEN_SIZE * 2)

        # Normalisation
        self.norm = nn.LayerNorm(HIDDEN_SIZE * 2)

        # Tête principale : décision trading
        self.decision_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, DENSE_SIZE),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(DENSE_SIZE, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3)   # VENTE / NEUTRE / ACHAT
        )

        # Tête confiance : score 0-100
        self.confidence_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Tête volatilité : anticiper l'ATR futur
        self.volatility_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

        # Tête sentiment / pressentiment marché
        self.sentiment_head = MarketSentimentHead(HIDDEN_SIZE * 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context      = self.attention(lstm_out)
        context      = self.norm(context)

        decision     = self.decision_head(context)
        confidence   = self.confidence_head(context)
        volatility   = self.volatility_head(context)
        sentiment    = self.sentiment_head(context)

        return decision, confidence, volatility, sentiment


# ============================================================
#  INDICATEURS TECHNIQUES — Arsenal complet
# ============================================================
def _ema(data, period):
    ema = np.zeros_like(data, dtype=np.float64)
    ema[0] = data[0]
    k = 2 / (period + 1)
    for i in range(1, len(data)):
        ema[i] = data[i] * k + ema[i-1] * (1 - k)
    return ema

def _normalize(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def _rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    ag = np.convolve(gain, np.ones(period)/period, 'same')
    al = np.convolve(loss, np.ones(period)/period, 'same')
    rs = np.where(al == 0, 100.0, ag / (al + 1e-10))
    return 100 - (100 / (1 + rs))

def _stochastic(high, low, close, k=14, d=3):
    stoch_k = np.zeros(len(close))
    for i in range(k, len(close)):
        hh = high[i-k:i].max()
        ll = low[i-k:i].min()
        stoch_k[i] = 100 * (close[i] - ll) / (hh - ll + 1e-10)
    stoch_d = np.convolve(stoch_k, np.ones(d)/d, 'same')
    return stoch_k, stoch_d

def _atr(high, low, close, period=14):
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    return np.convolve(tr, np.ones(period)/period, 'same')

def _bollinger(close, period=20, std_dev=2):
    mid  = np.convolve(close, np.ones(period)/period, 'same')
    std  = np.array([close[max(0,i-period):i].std() for i in range(len(close))])
    return mid - std_dev*std, mid, mid + std_dev*std

def _williams_r(high, low, close, period=14):
    wr = np.zeros(len(close))
    for i in range(period, len(close)):
        hh = high[i-period:i].max()
        ll = low[i-period:i].min()
        wr[i] = -100 * (hh - close[i]) / (hh - ll + 1e-10)
    return wr

def _cci(high, low, close, period=20):
    tp  = (high + low + close) / 3
    sma = np.convolve(tp, np.ones(period)/period, 'same')
    mad = np.array([np.abs(tp[max(0,i-period):i] - sma[i]).mean()
                    for i in range(len(tp))])
    return (tp - sma) / (0.015 * mad + 1e-10)

def _mfi(high, low, close, vol, period=14):
    tp   = (high + low + close) / 3
    rmf  = tp * vol
    pos  = np.where(np.diff(tp, prepend=tp[0]) > 0, rmf, 0)
    neg  = np.where(np.diff(tp, prepend=tp[0]) < 0, rmf, 0)
    pmf  = np.convolve(pos, np.ones(period)/period, 'same')
    nmf  = np.convolve(neg, np.ones(period)/period, 'same')
    return 100 - (100 / (1 + pmf / (nmf + 1e-10)))

def _obv(close, vol):
    direction = np.sign(np.diff(close, prepend=close[0]))
    return np.cumsum(direction * vol)

def _vwap(high, low, close, vol):
    tp  = (high + low + close) / 3
    return np.cumsum(tp * vol) / (np.cumsum(vol) + 1e-10)

def _momentum(close, period=10):
    mom = np.zeros_like(close)
    mom[period:] = close[period:] - close[:-period]
    return mom

def _roc(close, period=12):
    roc = np.zeros_like(close)
    roc[period:] = (close[period:] - close[:-period]) / (close[:-period] + 1e-10) * 100
    return roc

def _adx(high, low, close, period=14):
    plus_dm  = np.where((high - np.roll(high,1)) > (np.roll(low,1) - low),
                         np.maximum(high - np.roll(high,1), 0), 0)
    minus_dm = np.where((np.roll(low,1) - low) > (high - np.roll(high,1)),
                         np.maximum(np.roll(low,1) - low, 0), 0)
    tr       = np.maximum(high - low,
               np.maximum(np.abs(high - np.roll(close,1)),
                          np.abs(low  - np.roll(close,1))))
    atr14  = np.convolve(tr,       np.ones(period)/period, 'same')
    pdi    = 100 * np.convolve(plus_dm,  np.ones(period)/period, 'same') / (atr14 + 1e-10)
    mdi    = 100 * np.convolve(minus_dm, np.ones(period)/period, 'same') / (atr14 + 1e-10)
    dx     = 100 * np.abs(pdi - mdi) / (pdi + mdi + 1e-10)
    return np.convolve(dx, np.ones(period)/period, 'same'), pdi, mdi

def _ichimoku(high, low):
    tenkan  = (np.convolve(high, np.ones(9)/9,  'same') +
               np.convolve(low,  np.ones(9)/9,  'same')) / 2
    kijun   = (np.convolve(high, np.ones(26)/26, 'same') +
               np.convolve(low,  np.ones(26)/26, 'same')) / 2
    senkou_a = (tenkan + kijun) / 2
    return tenkan, kijun, senkou_a

def _fractal_dimension(close, period=20):
    fd = np.zeros(len(close))
    for i in range(period, len(close)):
        seg = close[i-period:i]
        n   = len(seg)
        if seg.max() - seg.min() < 1e-10:
            fd[i] = 1.0
            continue
        hurst = np.log(n) / np.log(n / (np.ptp(seg) / (seg.std() + 1e-10) + 1))
        fd[i] = max(1.0, min(2.0, 2 - hurst))
    return fd

def _zscore(close, period=20):
    zs = np.zeros_like(close)
    for i in range(period, len(close)):
        seg    = close[i-period:i]
        zs[i]  = (close[i] - seg.mean()) / (seg.std() + 1e-10)
    return zs

def _volume_profile(close, vol, bins=20):
    hist, edges = np.histogram(close, bins=bins, weights=vol)
    poc_idx     = np.argmax(hist)
    poc_price   = (edges[poc_idx] + edges[poc_idx+1]) / 2
    poc_arr     = np.full(len(close), poc_price)
    return _normalize(poc_arr)

def _detect_candle_patterns(open_, high, low, close):
    """Détecte : doji, engulfing haussier/baissier, marteau, étoile filante."""
    patterns = np.zeros(len(close))
    body     = np.abs(close - open_)
    wick_up  = high - np.maximum(close, open_)
    wick_dn  = np.minimum(close, open_) - low
    for i in range(1, len(close)):
        # Doji
        if body[i] < 0.1 * (high[i] - low[i] + 1e-10):
            patterns[i] = 0.5
        # Marteau (bullish)
        elif wick_dn[i] > 2 * body[i] and wick_up[i] < body[i]:
            patterns[i] = 1.0
        # Étoile filante (bearish)
        elif wick_up[i] > 2 * body[i] and wick_dn[i] < body[i]:
            patterns[i] = -1.0
        # Engulfing haussier
        elif (close[i] > open_[i] and close[i-1] < open_[i-1]
              and close[i] > open_[i-1] and open_[i] < close[i-1]):
            patterns[i] = 2.0
        # Engulfing baissier
        elif (close[i] < open_[i] and close[i-1] > open_[i-1]
              and close[i] < open_[i-1] and open_[i] > close[i-1]):
            patterns[i] = -2.0
    return _normalize(patterns + 2)  # ramener en [0,1]

def _support_resistance(close, period=50):
    sr = np.zeros_like(close)
    for i in range(period, len(close)):
        seg     = close[i-period:i]
        support = seg.min()
        resist  = seg.max()
        sr[i]   = (close[i] - support) / (resist - support + 1e-10)
    return sr


# ============================================================
#  CONSTRUCTION DES FEATURES (40 dimensions)
# ============================================================
def build_features(data):
    close  = data['Close'].values.flatten().astype(np.float64)
    high   = data['High'].values.flatten().astype(np.float64)
    low    = data['Low'].values.flatten().astype(np.float64)
    vol    = data['Volume'].values.flatten().astype(np.float64)
    open_  = data['Open'].values.flatten().astype(np.float64)

    rsi              = _rsi(close)
    ema9             = _ema(close, 9)
    ema21            = _ema(close, 21)
    ema50            = _ema(close, 50)
    ema200           = _ema(close, 200)
    macd             = _ema(close, 12) - _ema(close, 26)
    macd_signal      = _ema(macd, 9)
    macd_hist        = macd - macd_signal
    stoch_k, stoch_d = _stochastic(high, low, close)
    atr              = _atr(high, low, close)
    bb_low, bb_mid, bb_high = _bollinger(close)
    bb_width         = bb_high - bb_low
    bb_pos           = (close - bb_low) / (bb_width + 1e-10)
    wr               = _williams_r(high, low, close)
    cci              = _cci(high, low, close)
    mfi              = _mfi(high, low, close, vol)
    obv              = _obv(close, vol)
    vwap             = _vwap(high, low, close, vol)
    momentum         = _momentum(close)
    roc              = _roc(close)
    adx, pdi, mdi   = _adx(high, low, close)
    tenkan, kijun, senkou = _ichimoku(high, low)
    fd               = _fractal_dimension(close)
    zscore           = _zscore(close)
    vp               = _volume_profile(close, vol)
    patterns         = _detect_candle_patterns(open_, high, low, close)
    sr               = _support_resistance(close)

    # Divergences RSI/Prix
    rsi_div  = _normalize(rsi - np.roll(rsi, 5))
    # Volatilité normalisée
    vol_norm = _normalize(vol)
    # Spread HL normalisé
    spread   = _normalize(high - low)
    # Prix vs VWAP
    vs_vwap  = _normalize(close - vwap)
    # EMA croisements
    ema_cross_9_21  = _normalize(ema9 - ema21)
    ema_cross_21_50 = _normalize(ema21 - ema50)
    ema_cross_50_200= _normalize(ema50 - ema200)

    features = np.stack([
        _normalize(close),          # 0  Prix brut
        _normalize(rsi),            # 1  RSI
        _normalize(macd),           # 2  MACD
        _normalize(macd_signal),    # 3  Signal MACD
        _normalize(macd_hist),      # 4  Histogramme MACD
        _normalize(stoch_k),        # 5  Stochastique K
        _normalize(stoch_d),        # 6  Stochastique D
        _normalize(atr),            # 7  ATR volatilité
        bb_pos,                     # 8  Position Bollinger
        _normalize(bb_width),       # 9  Largeur Bollinger
        _normalize(wr),             # 10 Williams %R
        _normalize(cci),            # 11 CCI
        _normalize(mfi),            # 12 Money Flow Index
        _normalize(obv),            # 13 OBV
        vs_vwap,                    # 14 vs VWAP
        _normalize(momentum),       # 15 Momentum
        _normalize(roc),            # 16 Rate of Change
        _normalize(adx),            # 17 ADX force tendance
        _normalize(pdi),            # 18 +DI
        _normalize(mdi),            # 19 -DI
        _normalize(tenkan),         # 20 Ichimoku Tenkan
        _normalize(kijun),          # 21 Ichimoku Kijun
        _normalize(senkou),         # 22 Ichimoku Senkou A
        _normalize(fd),             # 23 Dimension fractale
        _normalize(zscore),         # 24 Z-Score
        vp,                         # 25 Volume Profile (POC)
        patterns,                   # 26 Patterns bougies
        sr,                         # 27 Support/Résistance
        rsi_div,                    # 28 Divergence RSI
        vol_norm,                   # 29 Volume normalisé
        spread,                     # 30 Spread H-L
        ema_cross_9_21,             # 31 Croisement EMA 9/21
        ema_cross_21_50,            # 32 Croisement EMA 21/50
        ema_cross_50_200,           # 33 Croisement EMA 50/200
        _normalize(ema9),           # 34 EMA 9
        _normalize(ema21),          # 35 EMA 21
        _normalize(ema50),          # 36 EMA 50
        _normalize(ema200),         # 37 EMA 200
        _normalize(bb_high),        # 38 Bollinger haut
        _normalize(bb_low),         # 39 Bollinger bas
    ], axis=1)

    return np.nan_to_num(features, nan=0.0), close, atr


# ============================================================
#  SÉQUENCES + LABELS AVANCÉS
# ============================================================
def build_sequences(features, close, atr):
    X, y = [], []
    for i in range(SEQ_LEN, len(features) - 3):
        X.append(features[i-SEQ_LEN:i])
        future_return = (close[i+1] - close[i]) / (close[i] + 1e-10)
        threshold     = 0.0003 + atr[i] * 0.0001  # seuil adaptatif à la volatilité
        if future_return > threshold:
            label = 2   # ACHAT
        elif future_return < -threshold:
            label = 0   # VENTE
        else:
            label = 1   # NEUTRE
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ============================================================
#  MÉMOIRE CUMULATIVE
# ============================================================
def load_memory():
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, 'r') as f:
            return json.load(f)
    return {"sessions": 0, "total_epochs": 0, "best_loss": 9999,
            "win_rate": 0.0, "signals": [], "anomalies": []}

def save_memory(mem):
    with open(MEMORY_PATH, 'w') as f:
        json.dump(mem, f, indent=2)

def log_trade(signal, price, sl, tp, confidence, sentiment_label):
    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            log = json.load(f)
    log.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "signal":    signal,
        "price":     float(price),
        "sl":        float(sl),
        "tp":        float(tp),
        "confidence": float(confidence),
        "sentiment": sentiment_label
    })
    with open(LOG_PATH, 'w') as f:
        json.dump(log[-500:], f, indent=2)  # garder les 500 derniers


# ============================================================
#  TELEGRAM
# ============================================================
def send_telegram(message):
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=5)
    except Exception:
        pass


# ============================================================
#  ENTRAÎNEMENT — Mémoire cumulative + Early Stopping
# ============================================================
def train_bot(epochs=300, update_callback=None):
    mem = load_memory()

    if update_callback:
        update_callback(f"📡 Session #{mem['sessions']+1} — Téléchargement XAUUSD 1m...")

    data = yf.download(tickers="GC=F", period="7d", interval="1m", progress=False)
    data.dropna(inplace=True)

    features, close, atr = build_features(data)
    X, y = build_sequences(features, close, atr)

    split    = int(len(X) * 0.8)
    X_train  = torch.tensor(X[:split])
    y_train  = torch.tensor(y[:split])
    X_val    = torch.tensor(X[split:])
    y_val    = torch.tensor(y[split:])

    model = TradingLSTM()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        if update_callback:
            update_callback("🧠 Mémoire précédente chargée — enrichissement...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss   = mem.get('best_loss', 9999)
    patience    = 30
    no_improve  = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        dec, conf, vol_pred, sent = model(X_train)
        loss = criterion(dec, y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                v_dec, _, _, _ = model(X_val)
                v_loss = criterion(v_dec, y_val).item()
                preds  = v_dec.argmax(dim=1)
                acc    = (preds == y_val).float().mean().item()
            model.train()

            if update_callback:
                update_callback(
                    f"📈 Epoch {epoch+1}/{epochs} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Val: {v_loss:.4f} | "
                    f"Acc: {acc*100:.1f}%"
                )

            if v_loss < best_loss:
                best_loss  = v_loss
                no_improve = 0
                torch.save(model.state_dict(), MODEL_PATH)
            else:
                no_improve += 1
                if no_improve >= patience:
                    if update_callback:
                        update_callback(f"🛑 Early stopping — meilleure val_loss: {best_loss:.4f}")
                    break

    mem['sessions']     += 1
    mem['total_epochs'] += epoch + 1
    mem['best_loss']     = best_loss
    save_memory(mem)

    if update_callback:
        update_callback(
            f"✅ Entraînement terminé\n"
            f[colle le contenu complet ici]

# ============================================================
#  EXPORT ONNX — pour Android
# ============================================================
ONNX_PATH = "trading_model.onnx"

def export_onnx(model):
    model.eval()
    dummy = torch.zeros(1, SEQ_LEN, INPUT_SIZE)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"],
        output_names=["decision", "confidence", "volatility", "sentiment"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=17
    )
    return ONNX_PATH

# ============================================================
#  INFERENCE ONNX — Android (onnxruntime)
# ============================================================
def predict_onnx(features_seq):
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH)
    x = features_seq.astype(np.float32)[np.newaxis]
    dec, conf, vol, sent = sess.run(None, {"input": x})
    signal_map = {0: "🔴 VENTE", 1: "⚪ NEUTRE", 2: "🟢 ACHAT"}
    signal = signal_map[dec[0].argmax()]
    confidence = float(conf[0][0]) * 100
    return signal, confidence
    export_onnx(model)
