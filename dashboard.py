import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
from streamlit_folium import st_folium
import os
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beijing Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── AQI CONSTANTS ─────────────────────────────────────────────────────────────
AQI_COLORS = {
    'Good': '#00e400',
    'Moderate': '#cccc00',
    'Unhealthy for Sensitive Groups': '#ff7e00',
    'Unhealthy': '#ff0000',
    'Very Unhealthy': '#8f3f97',
    'Hazardous': '#7e0023'
}
AQI_ORDER = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups',
             'Unhealthy', 'Very Unhealthy', 'Hazardous']

STATION_COORDS = {
    'Aotizhongxin':  [39.9824, 116.3976],
    'Changping':     [40.2179, 116.2307],
    'Dingling':      [40.2908, 116.2197],
    'Dongsi':        [39.9298, 116.4172],
    'Guanyuan':      [39.9290, 116.3393],
    'Gucheng':       [39.9143, 116.1842],
    'Huairou':       [40.3281, 116.6294],
    'Nongzhanguan':  [39.9373, 116.4593],
    'Shunyi':        [40.1302, 116.6543],
    'Tiantan':       [39.8864, 116.4108],
    'Wanliu':        [39.9875, 116.2883],
    'Wanshouxigong': [39.8783, 116.3530]
}

MONTH_NAMES = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
MONTH_ORDER = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    main_data_path = os.path.join(os.path.dirname(__file__), "main_data.csv")
    if os.path.exists(main_data_path):
        df = pd.read_csv(main_data_path, parse_dates=['datetime'])
    else:
        # Fallback: load from raw data
        data_path = os.path.join(os.path.dirname(__file__), '..', 'Air-quality-dataset',
                                 'PRSA_Data_20130301-20170228')
        dfs = []
        for f in sorted(os.listdir(data_path)):
            if f.endswith('.csv'):
                dfs.append(pd.read_csv(os.path.join(data_path, f)))
        df = pd.concat(dfs, ignore_index=True)

        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        num_cols = ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM']
        for col in num_cols:
            df[col] = df.groupby('station')[col].transform(lambda x: x.fillna(x.median()))
        df['wd'] = df.groupby('station')['wd'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'N'))

        season_map = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
                      6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
        df['season'] = df['month'].map(season_map)
        df['month_name'] = df['month'].map(MONTH_NAMES)
        df['AQI_Category'] = df['PM2.5'].apply(classify_aqi)

    return df

def classify_aqi(pm25):
    if pd.isna(pm25): return 'Unknown'
    elif pm25 <= 12.0: return 'Good'
    elif pm25 <= 35.4: return 'Moderate'
    elif pm25 <= 55.4: return 'Unhealthy for Sensitive Groups'
    elif pm25 <= 150.4: return 'Unhealthy'
    elif pm25 <= 250.4: return 'Very Unhealthy'
    else: return 'Hazardous'

def get_marker_color(pm25):
    if pm25 <= 35.4: return '#00e400'
    elif pm25 <= 55.4: return '#ffff00'
    elif pm25 <= 75: return '#ff7e00'
    elif pm25 <= 100: return '#ff0000'
    else: return '#8f3f97'

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Flag_of_the_People%27s_Republic_of_China.svg/200px-Flag_of_the_People%27s_Republic_of_China.svg.png", width=60)
    st.title("🌫️ Beijing Air Quality")
    st.caption("Dashboard Analisis Kualitas Udara\nBeijing, 2013–2017")
    st.divider()

    df_raw = load_data()

    all_stations = sorted(df_raw['station'].unique())
    selected_stations = st.multiselect(
        "Pilih Stasiun",
        options=all_stations,
        default=all_stations,
        help="Filter berdasarkan stasiun monitoring"
    )

    all_years = sorted(df_raw['year'].unique())
    selected_years = st.multiselect(
        "Pilih Tahun",
        options=all_years,
        default=all_years,
        help="Filter berdasarkan tahun"
    )

    selected_seasons = st.multiselect(
        "Pilih Musim",
        options=SEASON_ORDER,
        default=SEASON_ORDER
    )

    st.divider()
    st.markdown("**Sumber Data:**")
    st.caption("PRSA Dataset – 12 stasiun monitoring Beijing")
    st.caption("Periode: Mar 2013 – Feb 2017")

# ── FILTER DATA ───────────────────────────────────────────────────────────────
df = df_raw[
    df_raw['station'].isin(selected_stations) &
    df_raw['year'].isin(selected_years) &
    df_raw['season'].isin(selected_seasons)
].copy()

if df.empty:
    st.error("Tidak ada data yang sesuai filter. Silakan ubah filter di sidebar.")
    st.stop()

# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.title("🌫️ Dashboard Analisis Kualitas Udara Beijing (2013–2017)")
st.markdown(
    "Dashboard ini menyajikan analisis interaktif kualitas udara di Beijing "
    "berdasarkan data dari **12 stasiun monitoring** selama periode **Maret 2013 – Februari 2017**."
)

# ── KPI METRICS ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

avg_pm25 = df['PM2.5'].mean()
max_pm25 = df['PM2.5'].max()
pct_unhealthy = (df['AQI_Category'].isin(['Unhealthy', 'Very Unhealthy', 'Hazardous'])).mean() * 100
pct_good = (df['AQI_Category'] == 'Good').mean() * 100
total_records = len(df)

with col1:
    st.metric("Rata-rata PM2.5", f"{avg_pm25:.1f} µg/m³",
              delta=f"{avg_pm25 - 5:.1f} di atas WHO" if avg_pm25 > 5 else None,
              delta_color="inverse")
with col2:
    st.metric("Maks PM2.5", f"{max_pm25:.0f} µg/m³")
with col3:
    st.metric("% Waktu Tidak Sehat", f"{pct_unhealthy:.1f}%",
              help="Persentase waktu dengan PM2.5 > 55.4 µg/m³")
with col4:
    st.metric("% Waktu Kualitas Baik", f"{pct_good:.1f}%",
              help="Persentase waktu dengan PM2.5 ≤ 12 µg/m³")
with col5:
    st.metric("Total Pengukuran", f"{total_records:,}")

st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Pertanyaan 1: Distribusi Stasiun & Pola Musiman",
    "🌤️ Pertanyaan 2: Faktor Meteorologi",
    "🗺️ Analisis Geospasial",
    "📈 Analisis Lanjutan (AQI Clustering)"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Q1: Station distribution + seasonal pattern
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Pertanyaan 1: Stasiun mana yang memiliki rata-rata PM2.5 tertinggi dan bagaimana pola musiman PM2.5?")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Rata-rata PM2.5 per Stasiun")
        station_avg = df.groupby('station')['PM2.5'].mean().sort_values()
        colors_s = ['#d73027' if v >= 90 else '#fc8d59' if v >= 75 else '#fee08b' if v >= 55 else '#91bfdb'
                    for v in station_avg.values]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(station_avg.index, station_avg.values, color=colors_s,
                       edgecolor='white', linewidth=0.8, height=0.7)
        for bar, val in zip(bars, station_avg.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', ha='left', fontsize=9, fontweight='bold')

        ax.axvline(x=35.4, color='#2c7bb6', linestyle='--', lw=1.5, alpha=0.8,
                   label='Moderate (35.4 µg/m³)')
        ax.axvline(x=75, color='#d7191c', linestyle='--', lw=1.5, alpha=0.8,
                   label='Unhealthy (75 µg/m³)')
        ax.set_xlabel('Rata-rata PM2.5 (µg/m³)', fontsize=10)
        ax.set_title('Rata-rata PM2.5 per Stasiun', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, station_avg.max() * 1.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("#### Pola Musiman PM2.5 per Bulan")
        monthly_avg = df.groupby('month_name')['PM2.5'].mean().reindex(MONTH_ORDER)
        colors_m = ['#d73027' if v >= 100 else '#fc8d59' if v >= 75 else '#fee08b' if v >= 55 else '#91bfdb'
                    for v in monthly_avg.values]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars2 = ax.bar(MONTH_ORDER, monthly_avg.values, color=colors_m,
                       edgecolor='white', linewidth=0.8, width=0.75)
        for bar, val in zip(bars2, monthly_avg.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        ax.axhline(y=35.4, color='#2c7bb6', linestyle='--', lw=1.5, alpha=0.8,
                   label='Moderate (35.4 µg/m³)')
        ax.axhline(y=75, color='#d7191c', linestyle='--', lw=1.5, alpha=0.8,
                   label='Unhealthy (75 µg/m³)')
        ax.axvspan(-0.4, 1.4, alpha=0.08, color='blue')
        ax.axvspan(8.6, 11.4, alpha=0.08, color='blue')
        ax.axvspan(5.6, 7.4, alpha=0.08, color='orange')
        ax.set_ylabel('Rata-rata PM2.5 (µg/m³)', fontsize=10)
        ax.set_title('Pola Musiman PM2.5 per Bulan', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, monthly_avg.max() * 1.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tren tahunan
    st.markdown("#### Tren PM2.5 Bulanan Sepanjang Waktu (Time Series)")
    yearly_monthly = df.groupby(['year', 'month'])['PM2.5'].mean().reset_index()
    yearly_monthly['date'] = pd.to_datetime(yearly_monthly[['year', 'month']].assign(day=1))
    yearly_monthly = yearly_monthly.sort_values('date')

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(yearly_monthly['date'], yearly_monthly['PM2.5'],
            color='#d73027', linewidth=1.5, alpha=0.8)
    ax.fill_between(yearly_monthly['date'], yearly_monthly['PM2.5'], alpha=0.15, color='#d73027')
    ax.axhline(y=75, color='orange', linestyle='--', lw=1, alpha=0.7, label='Batas Unhealthy')
    ax.axhline(y=35.4, color='blue', linestyle='--', lw=1, alpha=0.7, label='Batas Moderate')
    ax.set_ylabel('Rata-rata PM2.5 (µg/m³)', fontsize=10)
    ax.set_title('Tren Rata-rata PM2.5 Bulanan – Seluruh Stasiun Terpilih', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(
        "**Insight:** Gucheng memiliki rata-rata PM2.5 tertinggi, diikuti Dongsi dan Wanshouxigong. "
        "Pola musiman sangat jelas — PM2.5 di musim dingin (Des–Feb) 2–3x lebih tinggi dibanding "
        "musim panas (Jul–Agu). Seluruh stasiun memiliki rata-rata PM2.5 di atas batas 'Unhealthy'."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Q2: Meteorological factors
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Pertanyaan 2: Bagaimana pengaruh faktor meteorologi terhadap PM2.5?")

    col_a, col_b, col_c = st.columns(3)

    # Kecepatan angin
    with col_a:
        st.markdown("#### Kecepatan Angin vs PM2.5")
        wind_bins = [0, 1, 2, 4, 6, 100]
        wind_labels = ['0-1 m/s', '1-2 m/s', '2-4 m/s', '4-6 m/s', '>6 m/s']
        df['wind_cat'] = pd.cut(df['WSPM'], bins=wind_bins, labels=wind_labels)
        wind_pm25 = df.groupby('wind_cat', observed=True)['PM2.5'].mean()

        fig, ax = plt.subplots(figsize=(5, 4))
        colors_w = ['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4']
        bars = ax.bar(range(len(wind_pm25)), wind_pm25.values,
                      color=colors_w, edgecolor='white', width=0.7)
        ax.set_xticks(range(len(wind_pm25)))
        ax.set_xticklabels(wind_labels, fontsize=8, rotation=20)
        ax.set_ylabel('Rata-rata PM2.5 (µg/m³)', fontsize=9)
        ax.set_title('Pengaruh Kecepatan Angin', fontsize=10, fontweight='bold')
        for bar, val in zip(bars, wind_pm25.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        pct_wind = (wind_pm25.iloc[0] - wind_pm25.iloc[-1]) / wind_pm25.iloc[0] * 100
        st.metric("Penurunan PM2.5 (Tenang→Kencang)", f"{pct_wind:.0f}%")

    # Suhu
    with col_b:
        st.markdown("#### Suhu vs PM2.5")
        temp_bins = [-40, 0, 10, 20, 30, 50]
        temp_labels = ['<0°C', '0-10°C', '10-20°C', '20-30°C', '>30°C']
        df['temp_cat'] = pd.cut(df['TEMP'], bins=temp_bins, labels=temp_labels)
        temp_pm25 = df.groupby('temp_cat', observed=True)['PM2.5'].mean()

        fig, ax = plt.subplots(figsize=(5, 4))
        colors_t = ['#4575b4', '#74add1', '#e0f3f8', '#fdae61', '#d73027']
        bars = ax.bar(range(len(temp_pm25)), temp_pm25.values,
                      color=colors_t, edgecolor='white', width=0.7)
        ax.set_xticks(range(len(temp_pm25)))
        ax.set_xticklabels(temp_labels, fontsize=8)
        ax.set_ylabel('Rata-rata PM2.5 (µg/m³)', fontsize=9)
        ax.set_title('Pengaruh Suhu', fontsize=10, fontweight='bold')
        for bar, val in zip(bars, temp_pm25.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.metric("PM2.5 Tertinggi", f"Suhu <0°C ({temp_pm25.iloc[0]:.1f} µg/m³)")

    # Hujan
    with col_c:
        st.markdown("#### Curah Hujan vs PM2.5")
        df['rain_cond'] = df['RAIN'].apply(lambda x: 'Hujan' if x > 0 else 'Tidak Hujan')
        rain_pm25 = df.groupby('rain_cond')['PM2.5'].mean().reindex(['Hujan', 'Tidak Hujan'])

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(rain_pm25.index, rain_pm25.values,
                      color=['#4575b4', '#d73027'], edgecolor='white', width=0.5)
        ax.set_ylabel('Rata-rata PM2.5 (µg/m³)', fontsize=9)
        ax.set_title('Pengaruh Hujan', fontsize=10, fontweight='bold')
        for bar, val in zip(bars, rain_pm25.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        pct_rain = (rain_pm25['Tidak Hujan'] - rain_pm25['Hujan']) / rain_pm25['Tidak Hujan'] * 100
        st.metric("Penurunan PM2.5 saat Hujan", f"{pct_rain:.0f}%")

    # Correlation heatmap
    st.markdown("#### Matriks Korelasi: Faktor Meteorologi vs Polutan")
    corr_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, vmin=-1, vmax=1, ax=ax, mask=False,
                annot_kws={'size': 8}, linewidths=0.5)
    ax.set_title('Matriks Korelasi Antar Variabel', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(
        "**Insight:** Kecepatan angin (WSPM) berkorelasi negatif dengan PM2.5 — "
        "angin kencang mendispersikan polutan. Suhu rendah dan tekanan tinggi "
        "cenderung memerangkap polutan. Hujan menurunkan PM2.5 ~30-35%."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Geospatial
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Analisis Geospasial: Distribusi PM2.5 di Peta Beijing")
    st.markdown(
        "Peta interaktif berikut menampilkan rata-rata konsentrasi PM2.5 di setiap stasiun monitoring. "
        "**Ukuran lingkaran** proporsional dengan tingkat polusi. **Klik lingkaran** untuk detail."
    )

    station_geo = df.groupby('station').agg(
        avg_pm25=('PM2.5', 'mean'),
        max_pm25=('PM2.5', 'max'),
        pct_unhealthy=('PM2.5', lambda x: (x > 55.4).mean() * 100)
    ).round(2)

    m = folium.Map(location=[40.05, 116.37], zoom_start=10, tiles='CartoDB positron')

    for station, coords in STATION_COORDS.items():
        if station in station_geo.index:
            avg = station_geo.loc[station, 'avg_pm25']
            max_v = station_geo.loc[station, 'max_pm25']
            pct = station_geo.loc[station, 'pct_unhealthy']
            color = get_marker_color(avg)

            popup_html = f"""
            <div style='font-family: Arial; width: 210px'>
                <h4 style='margin: 0 0 5px 0; color: #222'>{station}</h4>
                <hr style='margin: 4px 0'>
                <b>Rata-rata PM2.5:</b> {avg:.1f} µg/m³<br>
                <b>Maks PM2.5:</b> {max_v:.0f} µg/m³<br>
                <b>% Waktu Tidak Sehat:</b> {pct:.1f}%
            </div>
            """

            folium.CircleMarker(
                location=coords,
                radius=max(8, avg / 7),
                color='white',
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=230),
                tooltip=f"{station}: {avg:.1f} µg/m³"
            ).add_to(m)

            folium.Marker(
                location=[coords[0] + 0.018, coords[1]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:9px;font-weight:bold;color:#222;white-space:nowrap">'
                         f'{station}</div>',
                    icon_size=(120, 20), icon_anchor=(60, 0)
                )
            ).add_to(m)

    st_folium(m, width=900, height=550)

    # Ranking table
    st.markdown("#### Peringkat Stasiun Berdasarkan Rata-rata PM2.5")
    rank_df = station_geo.sort_values('avg_pm25', ascending=False).reset_index()
    rank_df.columns = ['Stasiun', 'Rata-rata PM2.5 (µg/m³)', 'Maks PM2.5 (µg/m³)', '% Waktu Tidak Sehat']
    rank_df.insert(0, 'Peringkat', range(1, len(rank_df) + 1))

    def color_pm25(val):
        if isinstance(val, float):
            if val >= 90: return 'background-color: #ffcccc'
            elif val >= 75: return 'background-color: #ffe0cc'
            elif val >= 55: return 'background-color: #fff5cc'
            else: return 'background-color: #ccffcc'
        return ''

    styled = rank_df.style.applymap(color_pm25, subset=['Rata-rata PM2.5 (µg/m³)'])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.info(
        "**Insight Geospasial:** Stasiun di pusat kota (Gucheng, Dongsi, Wanshouxigong) "
        "memiliki PM2.5 tertinggi karena kepadatan industri dan lalu lintas. "
        "Stasiun di pinggiran (Dingling, Huairou) memiliki udara paling bersih berkat "
        "lokasi yang jauh dari sumber emisi dan lebih banyak ruang hijau."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – AQI Clustering
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Analisis Lanjutan: Clustering Kualitas Udara Berdasarkan Standar AQI")
    st.markdown(
        "Setiap pengukuran PM2.5 diklasifikasikan ke dalam kategori **AQI (Air Quality Index)** "
        "berdasarkan standar US EPA menggunakan metode **manual grouping (binning)**."
    )

    if 'AQI_Category' not in df.columns:
        df['AQI_Category'] = df['PM2.5'].apply(classify_aqi)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribusi Keseluruhan Kategori AQI")
        aqi_dist = df['AQI_Category'].value_counts().reindex(
            [c for c in AQI_ORDER if c in df['AQI_Category'].unique()])
        aqi_pct = (aqi_dist / len(df) * 100).round(2)

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(range(len(aqi_pct)), aqi_pct.values,
                      color=[AQI_COLORS.get(c, '#888') for c in aqi_pct.index],
                      edgecolor='white', linewidth=0.8, width=0.75)
        short_labels = ['Good', 'Moderate', 'USG', 'Unhealthy', 'Very\nUnhealthy', 'Hazardous']
        ax.set_xticks(range(len(aqi_pct)))
        ax.set_xticklabels([short_labels[AQI_ORDER.index(c)] for c in aqi_pct.index], fontsize=9)
        ax.set_ylabel('Persentase Waktu (%)', fontsize=10)
        ax.set_title('Distribusi Kategori AQI – Beijing 2013–2017', fontsize=11, fontweight='bold')
        for bar, val in zip(bars, aqi_pct.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Distribusi AQI per Stasiun")
        station_aqi = (df.groupby(['station', 'AQI_Category']).size()
                       .unstack(fill_value=0)
                       .reindex(columns=[c for c in AQI_ORDER if c in df['AQI_Category'].unique()]))
        station_aqi_pct = station_aqi.div(station_aqi.sum(axis=1), axis=0) * 100

        station_order_list = list(df.groupby('station')['PM2.5'].mean()
                                  .sort_values(ascending=False).index)
        station_aqi_pct = station_aqi_pct.reindex(
            [s for s in station_order_list if s in station_aqi_pct.index])

        fig, ax = plt.subplots(figsize=(7, 5))
        bottom = np.zeros(len(station_aqi_pct))
        for cat in station_aqi_pct.columns:
            ax.bar(range(len(station_aqi_pct)), station_aqi_pct[cat].values,
                   bottom=bottom, color=AQI_COLORS.get(cat, '#888'),
                   label=cat, edgecolor='white', linewidth=0.5, width=0.85)
            bottom += station_aqi_pct[cat].values

        ax.set_xticks(range(len(station_aqi_pct)))
        ax.set_xticklabels(station_aqi_pct.index, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Persentase Waktu (%)', fontsize=10)
        ax.set_title('Distribusi AQI per Stasiun\n(Tertinggi → Terendah PM2.5)', fontsize=11, fontweight='bold')
        ax.legend(title='Kategori AQI', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # AQI trend over time
    st.markdown("#### Tren Distribusi AQI per Bulan")
    df['month_year'] = df['datetime'].dt.to_period('M').astype(str)
    aqi_trend = (df.groupby(['month_year', 'AQI_Category']).size()
                 .unstack(fill_value=0)
                 .reindex(columns=[c for c in AQI_ORDER if c in df['AQI_Category'].unique()])
                 .apply(lambda x: x / x.sum() * 100, axis=1))

    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(len(aqi_trend))
    for cat in aqi_trend.columns:
        ax.bar(range(len(aqi_trend)), aqi_trend[cat].values, bottom=bottom,
               color=AQI_COLORS.get(cat, '#888'), label=cat, width=1.0)
        bottom += aqi_trend[cat].values

    xtick_positions = list(range(0, len(aqi_trend), 6))
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([aqi_trend.index[i] for i in xtick_positions], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Persentase (%)', fontsize=10)
    ax.set_title('Tren Distribusi Kategori AQI per Bulan', fontsize=11, fontweight='bold')
    ax.legend(title='AQI', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # AQI summary table
    st.markdown("#### Ringkasan Persentase Waktu per Kategori AQI per Stasiun")
    if 'station_aqi_pct' in dir():
        display_df = station_aqi_pct.round(1)
        st.dataframe(display_df.style.background_gradient(cmap='RdYlGn_r'),
                     use_container_width=True)

    st.info(
        "**Insight AQI Clustering:** Hanya ~7% waktu kualitas udara Beijing berada pada kategori 'Good'. "
        "Lebih dari 50% waktu berada pada kategori 'Unhealthy' atau lebih buruk. "
        "Distribusi AQI menunjukkan pola musiman yang kuat dengan kategori terburuk di musim dingin."
    )

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 12px'>"
    "Dashboard Proyek Analisis Data | Alfath Septyan | "
    "Sumber: PRSA Air Quality Dataset Beijing 2013-2017"
    "</div>",
    unsafe_allow_html=True
)
