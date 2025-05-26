from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import requests

app = Flask(__name__)

# --- Sector to company mapping (copied from Streamlit code) ---
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

def format_pct_change(entry, price):
    pct = ((price - entry) / entry) * 100
    return f"({abs(pct):.2f}%)"

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
    # Convert sets to sorted lists for JSON serialization
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
    # Data cleaning and validation
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
    # Detect signals
    detect_signals(df)
    # Prepare Plotly traces
    traces = []
    # Main price line
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
    # Tag markers
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
    # Layout
    last_date = df['date'].max()
    extended_date = last_date + timedelta(days=20)
    layout = {
        'height': 800,
        'width': 1800,
        'plot_bgcolor': "darkslategray",
        'paper_bgcolor': "darkslategray",
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
    return jsonify({'traces': traces, 'layout': layout, 'last_updated': str(last_updated)})

if __name__ == '__main__':
    app.run(debug=True)