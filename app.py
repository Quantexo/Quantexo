from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import requests

app = Flask(__name__)

# --- Sector to company mapping (copy from your Streamlit code) ---
sector_to_companies = {
    "Index": {"NEPSE"},
    "Sub-Index": {"BANKING", "DEVBANK", "FINANCE", "HOTELS", "HYDROPOWER", "INVESTMENT","LIFEINSU","MANUFACUTRE","MICROFINANCE","NONLIFEINSU", "OTHERS", "TRADING"},
    "Commercial Banks": {"ADBL","CZBIL","EBL","GBIME","HBL","KBL","LSL","MBL","NABIL","NBL","NICA","NIMB","NMB","PCBL","PRVU","SANIMA","SBI","SBL","SCB"},
    "Development Banks": {"CORBL","EDBL","GBBL","GRDBL","JBBL","KSBBL","LBBL","MDB","MLBL","MNBBL","NABBC","SADBL","SAPDBL","SHINE","SINDU"},
    "Finance": {"BFC","CFCL","GFCL","GMFIL","GUFL","ICFC","JFL","MFIL","MPFL","NFS","PFL","PROFL","RLFL","SFCL","SIFC"},
    "Hotels": {"CGH","CITY","KDL","OHL","SHL","TRH"},
    "Hydro Power": {"AHPC", "AHL", "AKJCL", "AKPL", "API", "BARUN", "BEDC", "BHDC", "BHPL", "BGWT", "BHL", "BNHC", "BPCL", "CHCL", "CHL", "CKHL", "DHPL", "DOLTI", "DORDI", "EHPL", "GHL", "GLH", "GVL", "HDHPC", "HHL", "HPPL", "HURJA", "IHL", "JOSHI", "KKHC", "KPCL", "KBSH", "LEC", "MAKAR", "MANDU", "MBJC", "MEHL", "MEL", "MEN", "MHCL", "MHNL", "MKHC", "MKHL", "MKJC", "MMKJL", "MHL", "MCHL", "MSHL", "NGPL", "NHDL", "NHPC", "NYADI", "PPL", "PHCL", "PMHPL", "PPCL", "RADHI", "RAWA", "RHGCL", "RFPL", "RIDI", "RHPL", "RURU", "SAHAS", "SHEL", "SGHC", "SHPC", "SIKLES", "SJCL", "SMH", "SMHL", "SMJC", "SPC", "SPDL", "SPHL", "SPL", "SSHL", "TAMOR", "TPC", "TSHL", "TVCL", "UHEWA", "ULHC", "UMHL", "UMRH", "UNHPL", "UPCL", "UPPER", "USHL", "USHEC", "VLUCL"},
    "Investment": {"CHDC","CIT","ENL","HATHY","HIDCL","NIFRA","NRN"},
    "Life Insurance":{"ALICL","CLI","CREST","GMLI","HLI","ILI","LICN","NLIC","NLICL","PMLI","RNLI","SJLIC","SNLI","SRLI"},
    "Manufacturing and Processing": {"BNL","BNT","GCIL","HDL","NLO","OMPL","SARBTM","SHIVM","SONA","UNL"},
    "Microfinance": {"ACLBSL","ALBSL","ANLB","AVYAN","CBBL","CYCL","DDBL","DLBS","FMDBL","FOWAD","GBLBS","GILB","GLBSL","GMFBS","HLBSL","ILBS","JBLB","JSLBB","KMCDB","LLBS","MATRI","MERO","MLBBL","MLBS","MLBSL","MSLB","NADEP","NESDO","NICLBSL","NMBMF","NMFBS","NMLBBL","NUBL","RSDC","SAMAJ","SHLB","SKBBL","SLBBL","SLBSL","SMATA","SMB","SMFBS","SMPDA","SWBBL","SWMF","ULBSL","UNLB","USLB","VLBS","WNLB"},
    "Non Life Insurance": {"HEI","IGI","NICL","NIL","NLG","NMIC","PRIN","RBCL","SALICO","SGIC"},
    "Others": {"HRL","MKCL","NRIC","NRM","NTC","NWCL"},
    "Trading": {"BBC","STC"}
}

def get_sheet_data(symbol):
    try:
        sheet_url = "https://docs.google.com/spreadsheets/d/1Q_En7VGGfifDmn5xuiF-t_02doPpwl4PLzxb4TBCW0Q/export?format=csv&gid=0"
        df = pd.read_csv(sheet_url)
        df = df.iloc[:, :7]
        df.columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        nepal_tz = pytz.timezone('Asia/Kathmandu')
        last_updated = datetime.now(nepal_tz)
        df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()
        filtered_df = df[df['symbol'].str.upper() == symbol.upper()]
        return filtered_df, last_updated
    except Exception as e:
        return pd.DataFrame(), None

def find_historical_resistance(df, current_price, swing_high, lookback_periods = 100):
    resistance_levels = []
    historical_data = df.tail(lookback_periods) if len(df) > lookback_periods else df
    potential_resistance = historical_data[historical_data['high'] > current_price]
    if not potential_resistance.empty:
        high_values = sorted(potential_resistance['high'].unique())
        cluster_threshold = current_price * 0.02
        clustered_levels = []
        current_cluster = []
        for high_val in high_values:
            if not current_cluster or (high_val - current_cluster[-1] <= cluster_threshold):
                current_cluster.append(high_val)
            else:
                clustered_levels.append(np.mean(current_cluster))
                current_cluster = [high_val]
        if current_cluster:
            clustered_levels.append(np.mean(current_cluster))
        resistance_levels.extend(clustered_levels)
    price_magnitude = 10 ** (len(str(int(current_price))) - 1)
    psychological_levels = []
    for multiplier in [1, 2, 5, 10, 15, 20, 25, 50]:
        level = price_magnitude * multiplier
        if level > current_price and level <= swing_high * 2:
            psychological_levels.append(level)
    resistance_levels.extend(psychological_levels)
    resistance_levels = sorted(list(set(resistance_levels)))
    return resistance_levels

def detect_seller_absorption(df, min_targets=3, max_targets=15):
    signals = []
    df['absorption'] = False
    df['entry_price'] = None
    df['stop_loss'] = None
    df['targets'] = None
    df['tag'] = df.get('tag', '')  # Ensure tag column exists

    df['avg_volume'] = df['volume'].rolling(20).mean()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    for i in range(2, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        if (prev['open'] > prev['close'] and
            prev['volume'] > prev['avg_volume'] * 2 and
            current['close'] > prev['open'] and
            current['volume'] > current['avg_volume'] * 2):
            recent_low = df['low'].iloc[max(0, i - 60):i].min()
            price_gain_pct = (current['close'] - recent_low) / recent_low
            if price_gain_pct > 0.60:
                continue
            if not df['absorption'].iloc[max(0, i-5):i].any():
                df.loc[df['tag'] == '🚀', 'tag'] = ''
                df.at[i, 'absorption'] = True
                df.at[i, 'tag'] = '🚀'
                entry = current['close']
                swing_high = df['high'].iloc[max(0,i-50):i].max()
                swing_low = df['low'].iloc[max(0,i-50):i].min()
                atr = df['atr'].iloc[i]
                max_sl_pct = entry * 0.92
                proposed_sl = min(swing_low, max_sl_pct)
                min_sl_pct = entry * 0.95
                stop_loss = max(proposed_sl, min_sl_pct)
                targets = []
                resistance_levels = find_historical_resistance(df[:i], entry, swing_high)
                nearby_resistance = [r for r in resistance_levels if (r > entry) and (r <= entry * 1.20)]
                if nearby_resistance:
                    targets.extend(nearby_resistance[:max_targets])
                if len(targets) < max_targets:
                    fib_levels = [0.25, 0.33, 0.44, 0.65]
                    price_range = max(atr * 3, entry * 0.10)
                    fib_targets = [entry + (price_range * level) for level in fib_levels]
                    targets.extend(fib_targets)
                if len(targets) < max_targets:
                    pct_targets = [
                        entry * 1.05,
                        entry * 1.10,
                        entry * 1.15,
                        entry * 1.20
                    ]
                    targets.extend(pct_targets)
                targets = sorted(list(set([t for t in targets if t > entry])))
                targets = targets[:max_targets]
                conservative_entries = [
                    entry * 0.99,
                    entry * 0.98,
                    stop_loss * 1.01
                ]
                hit_dates = [None] * len(targets)
                signals.append({
                    'date': current['date'],
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'targets': targets,
                    'conservative_entries': conservative_entries,
                    'hit_dates': hit_dates,
                    'hit_stop': False,
                    'hit_targets': [False] * len(targets),
                    'is_current_signal': True
                })
                for prev_signal in signals[:-1]:
                    prev_signal['is_current_signal'] = False
                df.at[i, 'entry_price'] = entry
                df.at[i, 'stop_loss'] = stop_loss
                df.at[i, 'targets'] = targets
    for signal in signals:
        subsequent_data = df[df['date'] > signal['date']]
        stop_hits = subsequent_data[subsequent_data['low'] <= signal['stop_loss']]
        if not stop_hits.empty:
            signal['hit_stop'] = True
            signal['stop_hit_date'] = stop_hits.iloc[0]['date']
        subsequent_data = subsequent_data.head(20)
        for i, target in enumerate(signal['targets']):
            target_hits = subsequent_data[subsequent_data['high'] >= target]
            if not target_hits.empty:
                signal['hit_targets'][i] = True
                signal['hit_dates'][i] = target_hits.iloc[0]['date']
    return df, signals

def format_pct_change(entry, price):
    pct = ((price - entry) / entry) * 100
    return f"({abs(pct):.2f}%)"

def get_absorption_summary_annotation(signals):
    table_content = ["<b>SELLER ABSORPTION TRADE</b>"]
    current_signal = None
    for signal in signals:
        if signal.get('is_current_signal', False):
            current_signal = signal
            break
    if current_signal is None and signals:
        current_signal = max(signals, key=lambda x: x['date'])
    def fmt_date(dt):
        if isinstance(dt, pd.Timestamp):
            return dt.strftime('%b %d, %Y')
        elif isinstance(dt, datetime):
            return dt.strftime('%b %d, %Y')
        elif isinstance(dt, str):
            try:
                return pd.to_datetime(dt).strftime('%b %d, %Y')
            except:
                return str(dt)
        return str(dt)
    if current_signal:
        trade_status = ""
        if current_signal['hit_stop']:
            trade_status = f" [STOPPED OUT on {fmt_date(current_signal.get('stop_hit_date'))}]"
        elif any(current_signal['hit_targets']):
            hit_count = sum(current_signal['hit_targets'])
            trade_status = f" [ACTIVE - {hit_count} targets hit]"
        else:
            trade_status = " [ACTIVE]"
        table_content.extend([
            f"<b>Aggressive Entry</b> = {current_signal['entry']:.2f} ({fmt_date(current_signal['date'])}){trade_status}",
            f"<b>Conservative Entry</b> = {current_signal['conservative_entries'][0]:.2f}, {current_signal['conservative_entries'][1]:.2f}, {current_signal['conservative_entries'][2]:.2f}"
        ])
        targets_text = []
        for i, (target, hit_date) in enumerate(zip(current_signal['targets'], current_signal['hit_dates'])):
            status = f"✅ HIT on {fmt_date(hit_date)}" if hit_date else "⏳ PENDING"
            pct = format_pct_change(current_signal['entry'], target)
            targets_text.append(f"- TP {i+1} = {target:.2f} {pct} {status}")
        sl_pct = format_pct_change(current_signal['entry'], current_signal['stop_loss'])
        stop_status = f"❌ HIT on {fmt_date(current_signal.get('stop_hit_date'))}" if current_signal['hit_stop'] else "⏳ ACTIVE"
        table_content.extend(targets_text + ["", f"<b>Stop Loss</b> = {current_signal['stop_loss']:.2f} {sl_pct} {stop_status}"])
    else:
        table_content.append("No seller absorption trades found.")
    return {
        'xref': "paper", 'yref': "paper",
        'x': 0.03, 'y': 0.97,
        'text': "<br>".join(table_content),
        'showarrow': False,
        'align': "left",
        'bgcolor': "rgba(0,0,0,0)",
        'font': {'color': "white", 'size': 12, 'family': "Courier New, monospace"}
    }

def detect_signals(df):
    results = []
    df['point_change'] = df['close'].diff().fillna(0)
    df['tag'] = ''
    min_window = min(20, max(5, len(df) // 2)) 
    avg_volume = df['volume'].rolling(window=min_window).mean().fillna(method='bfill').fillna(df['volume'].mean())
    for i in range(min(3, len(df)-1), len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        next_candles = df.iloc[i + 1:min(i + 6, len(df))]
        body = abs(row['close'] - row['open'])
        prev_body = abs(prev['close'] - prev['open'])
        recent_tags = df['tag'].iloc[max(0, i - 9):i]
        if (
            row['close'] > row['open'] and
            row['volume'] > avg_volume[i] * 1.2
        ):
            df.loc[df['tag'] == '⛔', 'tag'] = ''
            for j, candle in next_candles.iterrows():
                if candle['close'] < row['open']:
                    df.at[j, 'tag'] = '⛔'
                    break
        if (
            row['close'] > row['open'] and
            row['close'] >= row['high'] - (row['high'] - row['low']) * 0.1 and
            row['volume'] > avg_volume[i] * 2 and
            body > prev_body and
            '🟢' not in recent_tags.values
        ):
            df.at[i, 'tag'] = '🟢'
        if (
            row['open'] > row['close'] and
            row['close'] <= row['low'] + (row['high'] - row['low']) * 0.1 and
            row['volume'] > avg_volume[i] * 2 and
            body > prev_body and
            '🔴' not in recent_tags.values
        ):
            df.at[i, 'tag'] = '🔴'
        if (
            i >= 10 and
            row['close'] > max(df['high'].iloc[i - 10:i]) and
            row['volume'] > avg_volume[i] * 1.8
        ):
            if not (df['tag'].iloc[i - 8:i] == '💥').any():
                df.at[i, 'tag'] = '💥'
        if (
            i >= 10 and
            row['close'] < min(df['low'].iloc[i - 10:i]) and
            row['volume'] > avg_volume[i] * 1.8
        ):
            if not (df['tag'].iloc[i - 8:i] == '💣').any():
                df.at[i, 'tag'] = '💣'
        if (
            row['close'] > row['open'] and
            body > (row['high'] - row['low']) * 0.85 and
            row['volume'] > avg_volume[i] * 2
        ):
            df.at[i, 'tag'] = '🐂'
        if (
            row['open'] > row['close'] and
            body > (row['high'] - row['low']) * 0.85 and
            row['volume'] > avg_volume[i] * 2
        ):
            df.at[i, 'tag'] = '🐻'
        if df.at[i, 'tag']:
            results.append({
                'symbol': row['symbol'],
                'tag': df.at[i, 'tag'],
                'date': row['date'] if isinstance(row['date'], str) else row['date'].strftime('%Y-%m-%d')
            })
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sectors')
def sectors():
    return jsonify({k: sorted(list(v)) for k, v in sector_to_companies.items()})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    symbol = data.get('symbol', '').strip().upper()
    if not symbol:
        return jsonify({'error': 'No symbol provided.'}), 400
    df, last_updated = get_sheet_data(symbol)
    if df.empty:
        return jsonify({'error': f'No data found for {symbol}'}), 404
    df.columns = [col.lower() for col in df.columns]
    required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(set(df.columns)):
        return jsonify({'error': 'Missing required columns.'}), 400
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isnull().any():
        return jsonify({'error': 'Invalid date format in some rows.'}), 400
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^\\d.]', '', regex=True), errors='coerce')
        if df[col].isnull().any():
            return jsonify({'error': f'Invalid values in {col} column.'}), 400
    df = df.dropna()
    if len(df) == 0:
        return jsonify({'error': 'No valid data after cleaning.'}), 400
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    detect_signals(df)
    df, all_absorptions = detect_seller_absorption(df)
    traces = []
    traces.append({
        'x': df['date'].dt.strftime('%Y-%m-%d').tolist(),
        'y': df['close'].tolist(),
        'mode': 'lines',
        'name': 'Close Price',
        'line': {'color': 'lightblue', 'width': 2},
        'customdata': df[['date', 'open', 'high', 'low', 'close', 'point_change']].values.tolist(),
        'hovertemplate': (
            "📅 Date: %{customdata[0]}<br>" +
            "🟢 Open: %{customdata[1]:.2f}<br>" +
            "📈 High: %{customdata[2]:.2f}<br>" +
            "📉 Low: %{customdata[3]:.2f}<br>" +
            "🔚 LTP: %{customdata[4]:.2f}<br>" +
            "📊 Point Change: %{customdata[5]:.2f}<extra></extra>"
        )
    })
    tag_labels = {
        '🟢': '🟢 Aggressive Buyers',
        '🔴': '🔴 Aggressive Sellers',
        '⛔': '⛔ Buyer Absorption',
        '🚀': '🚀 Seller Absorption',
        '💥': '💥 Bullish POR',
        '💣': '💣 Bearish POR',
        '🐂': '🐂 Bullish POI',
        '🐻': '🐻 Bearish POI'
    }
    signals = df[df['tag'] != '']
    for tag in signals['tag'].unique():
        subset = signals[signals['tag'] == tag]
        traces.append({
            'x': subset['date'].dt.strftime('%Y-%m-%d').tolist(),
            'y': subset['close'].tolist(),
            'mode': 'markers+text',
            'name': tag_labels.get(tag, tag),
            'text': [tag] * len(subset),
            'textposition': 'top center',
            'textfont': {'size': 20},
            'marker': {'size': 14, 'symbol': "circle", 'color': 'white'},
            'customdata': subset[['open', 'high', 'low', 'close', 'point_change']].values.tolist(),
            'hovertemplate': (
                "📅 Date: %{x}<br>" +
                "🟢 Open: %{customdata[0]:.2f}<br>" +
                "📈 High: %{customdata[1]:.2f}<br>" +
                "📉 Low: %{customdata[2]:.2f}<br>" +
                "🔚 LTP: %{customdata[3]:.2f}<br>" +
                "📊 Point Change: %{customdata[4]:.2f}<br>" +
                f"{tag_labels.get(tag, tag)}<extra></extra>"
            )
        })
    last_date = df['date'].max()
    extended_date = last_date + timedelta(days=20)
    layout = {
        'autosize': True,
        'plot_bgcolor': "#2F4F4F",
        'paper_bgcolor': "#2F4F4F",
        'font': {'color': "white"},
        'xaxis': {
            'title': "Date",
            'tickangle': -45,
            'showgrid': False,
            'range': [
                (df['date'].max() - pd.Timedelta(days=365)).strftime('%Y-%m-%d'),
                extended_date.strftime('%Y-%m-%d')
            ],
            'rangeselector': {
                'buttons': [
                    {'count': 3, 'label': "3m", 'step': "month", 'stepmode': "backward"},
                    {'count': 6, 'label': "6m", 'step': "month", 'stepmode': "backward"},
                    {'count': 1, 'label': "YTD", 'step': "year", 'stepmode': "todate"},
                    {'count': 1, 'label': "1y", 'step': "year", 'stepmode': "backward"},
                    {'count': 2, 'label': "2y", 'step': "year", 'stepmode': "backward"},
                    {'step': "all"}
                ]
            }
        },
        'yaxis': {
            'title': "Price",
            'showgrid': False,
            'zeroline': True,
            'zerolinecolor': "gray",
            'autorange': True
        },
        'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
        'legend': {
            'orientation': "h",
            'yanchor': "top",
            'y': -0.12,
            'xanchor': "center",
            'x': 0.5,
            'font': {'size': 14},
            'bgcolor': "rgba(0,0,0,0)"
        },
        'dragmode': "zoom",
        'annotations': [
            {
                'xref': "paper", 'yref': "paper",
                'x': 0.5, 'y': 0.5,
                'text': f"Quantexo<br>{symbol}",
                'showarrow': False,
                'font': {'size': 40, 'color': "rgba(128,128,128,0.2)"},
                'align': "center"
            }
        ]
    }
    # Add absorption summary annotation if available
    if all_absorptions:
        layout['annotations'].append(get_absorption_summary_annotation(all_absorptions))
    return jsonify({'traces': traces, 'layout': layout, 'last_updated': str(last_updated)})

if __name__ == '__main__':
    app.run(debug=True)