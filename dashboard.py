import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Beijing Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
AQI_COLORS = {
    'Good': '#00e400',
    'Moderate': '#cccc00',
    'Unhealthy for Sensitive Groups': '#ff7e00',
    'Unhealthy': '#ff0000',
    'Very Unhealthy': '#8f3f97',
    'Hazardous': '#7e0023'
}
AQI_ORDER = [
    'Good', 'Moderate', 'Unhealthy for Sensitive Groups',
    'Unhealthy', 'Very Unhealthy', 'Hazardous'
]
STATION_COORDS = {
    'Aotizhongxin': [39.9824, 116.3976], 'Changping': [40.2179, 116.2307],
    'Dingling': [40.2908, 116.2197],     'Dongsi': [39.9298, 116.4172],
    'Guanyuan': [39.9290, 116.3393],     'Gucheng': [39.9143, 116.1842],
    'Huairou': [40.3281, 116.6294],      'Nongzhanguan': [39.9373, 116.4593],
    'Shunyi': [40.1302, 116.6543],       'Tiantan': [39.8864, 116.4108],
    'Wanliu': [39.9875, 116.2883],       'Wanshouxigong': [39.8783, 116.3530]
}
MONTH_ORDER = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
MONTH_MAP = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
             7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']
SEASON_MAP = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
              6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}

BASE_CLR   = '#aed6f1'   # light blue – default bar color
HI_CLR     = '#1a5276'   # dark blue  – highlight (max/worst)
GOOD_CLR   = '#2ecc71'   # green      – best condition


def classify_aqi(pm25):
    if pd.isna(pm25): return 'Unknown'
    elif pm25 <= 12.0: return 'Good'
    elif pm25 <= 35.4: return 'Moderate'
    elif pm25 <= 55.4: return 'Unhealthy for Sensitive Groups'
    elif pm25 <= 150.4: return 'Unhealthy'
    elif pm25 <= 250.4: return 'Very Unhealthy'
    else: return 'Hazardous'


# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    main_csv = os.path.join(os.path.dirname(__file__), 'main_data.csv')
    if os.path.exists(main_csv):
        df = pd.read_csv(main_csv)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    else:
        # fallback: load raw data
        raw_path = os.path.join(os.path.dirname(__file__), '..', 'data')
        if not os.path.exists(raw_path):
            raw_path = os.path.join(os.path.dirname(__file__), '..', 'Air-quality-dataset',
                                    'PRSA_Data_20130301-20170228')
        files = sorted([f for f in os.listdir(raw_path) if f.endswith('.csv')])
        dfs = [pd.read_csv(os.path.join(raw_path, f)) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        num_cols = ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM']
        for col in num_cols:
            df[col] = df.groupby('station')[col].transform(lambda x: x.fillna(x.median()))
        df['wd'] = df.groupby('station')['wd'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'N'))

    # ensure derived columns
    if 'season' not in df.columns:
        df['season'] = df['month'].map(SEASON_MAP)
    if 'month_name' not in df.columns:
        df['month_name'] = df['month'].map(MONTH_MAP)
    if 'AQI_Category' not in df.columns:
        df['AQI_Category'] = df['PM2.5'].apply(classify_aqi)

    return df


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌫️ Beijing Air Quality")
    st.caption("Dashboard Analisis Kualitas Udara\nBeijing, 2013–2017")
    st.divider()

    df_raw = load_data()

    selected_stations = st.multiselect(
        "Pilih Stasiun", options=sorted(df_raw['station'].unique()),
        default=sorted(df_raw['station'].unique()))
    selected_years = st.multiselect(
        "Pilih Tahun", options=sorted(df_raw['year'].unique()),
        default=sorted(df_raw['year'].unique()))
    selected_seasons = st.multiselect(
        "Pilih Musim", options=SEASON_ORDER, default=SEASON_ORDER)

    st.divider()
    st.caption("Sumber: PRSA Dataset, Mar 2013 – Feb 2017")

# ── FILTER ────────────────────────────────────────────────────────────────────
mask = (df_raw['station'].isin(selected_stations) &
        df_raw['year'].isin(selected_years) &
        df_raw['season'].isin(selected_seasons))
df = df_raw[mask].copy()

if df.empty:
    st.error("Tidak ada data yang sesuai filter.")
    st.stop()

# ── HEADER & KPI ─────────────────────────────────────────────────────────────
st.title("🌫️ Dashboard Analisis Kualitas Udara Beijing (2013–2017)")
st.markdown(
    "Analisis interaktif kualitas udara dari **12 stasiun monitoring** "
    "selama periode **Maret 2013 – Februari 2017**."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rata-rata PM2.5", f"{df['PM2.5'].mean():.1f} µg/m³")
c2.metric("Maks PM2.5", f"{df['PM2.5'].max():.0f} µg/m³")
c3.metric("% Waktu Tidak Sehat",
          f"{(df['AQI_Category'].isin(['Unhealthy','Very Unhealthy','Hazardous'])).mean()*100:.1f}%")
c4.metric("% Waktu Baik",
          f"{(df['AQI_Category'] == 'Good').mean()*100:.1f}%")
c5.metric("Total Pengukuran", f"{len(df):,}")
st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Pertanyaan 1: Distribusi & Pola Musiman",
    "🌤️ Pertanyaan 2: Faktor Meteorologi",
    "🗺️ Analisis Geospasial",
    "📈 Analisis Lanjutan (AQI Clustering)"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Pertanyaan 1: Stasiun mana yang memiliki rata-rata PM2.5 tertinggi "
                 "dan bagaimana pola musiman PM2.5 di Beijing?")

    col_l, col_r = st.columns(2)

    # ── PM2.5 per Stasiun ────────────────────────────────────────────────────
    with col_l:
        st.markdown("#### Rata-rata PM2.5 per Stasiun")
        station_avg = df.groupby('station')['PM2.5'].mean().sort_values()
        max_station = station_avg.idxmax()
        colors_s = [HI_CLR if s == max_station else BASE_CLR for s in station_avg.index]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(station_avg.index, station_avg.values,
                       color=colors_s, edgecolor='white', height=0.7)
        for bar, val in zip(bars, station_avg.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', fontsize=9, fontweight='bold')

        ax.axvline(35.4, color='#e67e22', ls='--', lw=1.5, alpha=0.8, label='Batas Moderate (35.4)')
        ax.axvline(75,   color='#e74c3c', ls='--', lw=1.5, alpha=0.8, label='Batas Unhealthy (75)')
        ax.set_xlabel('Rata-rata PM2.5 (µg/m³)')
        ax.set_title('Rata-rata PM2.5 per Stasiun\n(Warna gelap = stasiun tertinggi)',
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, station_avg.max() * 1.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Pola Musiman ─────────────────────────────────────────────────────────
    with col_r:
        st.markdown("#### Pola Musiman PM2.5 per Bulan")
        monthly_avg = df.groupby('month_name')['PM2.5'].mean().reindex(MONTH_ORDER)
        max_month = monthly_avg.idxmax()
        colors_m = [HI_CLR if m == max_month else BASE_CLR for m in MONTH_ORDER]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars2 = ax.bar(MONTH_ORDER, monthly_avg.values, color=colors_m,
                       edgecolor='white', width=0.75)
        for bar, val in zip(bars2, monthly_avg.fillna(0).values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f'{val:.0f}', ha='center', fontsize=8.5, fontweight='bold')

        ax.axhline(35.4, color='#e67e22', ls='--', lw=1.5, alpha=0.8, label='Moderate (35.4)')
        ax.axhline(75,   color='#e74c3c', ls='--', lw=1.5, alpha=0.8, label='Unhealthy (75)')
        ax.set_ylabel('Rata-rata PM2.5 (µg/m³)')
        ax.set_title(f'Pola Musiman PM2.5 per Bulan\n(Warna gelap = bulan tertinggi: {max_month})',
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, monthly_avg.max() * 1.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Time Series ──────────────────────────────────────────────────────────
    st.markdown("#### Tren PM2.5 Bulanan Sepanjang Waktu")
    ts = df.groupby(['year', 'month'])['PM2.5'].mean().reset_index()
    ts['date'] = pd.to_datetime(ts[['year', 'month']].assign(day=1))
    ts = ts.sort_values('date')

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ts['date'], ts['PM2.5'], color=HI_CLR, lw=1.8)
    ax.fill_between(ts['date'], ts['PM2.5'], alpha=0.15, color=HI_CLR)
    ax.axhline(75,   color='#e74c3c', ls='--', lw=1, alpha=0.7, label='Batas Unhealthy')
    ax.axhline(35.4, color='#e67e22', ls='--', lw=1, alpha=0.7, label='Batas Moderate')
    ax.set_ylabel('Rata-rata PM2.5 (µg/m³)')
    ax.set_title('Tren Rata-rata PM2.5 Bulanan', fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("**Insight:** Gucheng memiliki rata-rata PM2.5 tertinggi. Pola musiman jelas — "
            "PM2.5 di musim dingin (Des–Feb) 2–3x lebih tinggi dibanding musim panas (Jul–Agu).")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Pertanyaan 2: Bagaimana pengaruh faktor meteorologi terhadap PM2.5?")

    ca, cb, cc = st.columns(3)

    # ── Kecepatan Angin ───────────────────────────────────────────────────────
    with ca:
        st.markdown("#### Kecepatan Angin vs PM2.5")
        df_w = df.copy()
        df_w['wind_cat'] = pd.cut(df_w['WSPM'], bins=[0,1,2,4,6,100],
                                   labels=['0-1','1-2','2-4','4-6','>6'])
        wind_pm25 = df_w.groupby('wind_cat', observed=True)['PM2.5'].mean()

        max_w = wind_pm25.idxmax()
        colors_w = [HI_CLR if i == max_w else BASE_CLR for i in wind_pm25.index]

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(range(len(wind_pm25)), wind_pm25.values,
                      color=colors_w, edgecolor='white', width=0.7)
        ax.set_xticks(range(len(wind_pm25)))
        ax.set_xticklabels(['0-1\nm/s','1-2\nm/s','2-4\nm/s','4-6\nm/s','>6\nm/s'], fontsize=8)
        ax.set_ylabel('Rata-rata PM2.5 (µg/m³)')
        ax.set_title('Kecepatan Angin vs PM2.5\n(Gelap = kondisi terburuk)', fontweight='bold')
        for bar, val in zip(bars, wind_pm25.values):
            ax.text(bar.get_x() + bar.get_width()/2, val+1, f'{val:.0f}',
                    ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        pct = (wind_pm25.max() - wind_pm25.min()) / wind_pm25.max() * 100
        st.metric("Penurunan PM2.5 (lemah→kencang)", f"{pct:.0f}%")

    # ── Suhu ──────────────────────────────────────────────────────────────────
    with cb:
        st.markdown("#### Suhu vs PM2.5")
        df_t = df.copy()
        df_t['temp_cat'] = pd.cut(df_t['TEMP'], bins=[-40,0,10,20,30,50],
                                   labels=['<0°C','0-10°C','10-20°C','20-30°C','>30°C'])
        temp_pm25 = df_t.groupby('temp_cat', observed=True)['PM2.5'].mean()

        max_t = temp_pm25.idxmax()
        colors_t = [HI_CLR if i == max_t else BASE_CLR for i in temp_pm25.index]

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(range(len(temp_pm25)), temp_pm25.values,
                      color=colors_t, edgecolor='white', width=0.7)
        ax.set_xticks(range(len(temp_pm25)))
        ax.set_xticklabels(['<0°C','0-10°C','10-20°C','20-30°C','>30°C'], fontsize=8)
        ax.set_ylabel('Rata-rata PM2.5 (µg/m³)')
        ax.set_title('Suhu vs PM2.5\n(Gelap = kondisi terburuk)', fontweight='bold')
        for bar, val in zip(bars, temp_pm25.values):
            ax.text(bar.get_x() + bar.get_width()/2, val+1, f'{val:.0f}',
                    ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.metric("PM2.5 Tertinggi pada Suhu", f"<0°C ({temp_pm25.iloc[0]:.0f} µg/m³)")

    # ── Hujan ──────────────────────────────────────────────────────────────────
    with cc:
        st.markdown("#### Curah Hujan vs PM2.5")
        df_r = df.copy()
        df_r['rain_cond'] = df_r['RAIN'].apply(lambda x: 'Hujan' if x > 0 else 'Tidak Hujan')
        rain_pm25 = df_r.groupby('rain_cond')['PM2.5'].mean().reindex(['Hujan', 'Tidak Hujan'])

        colors_r = [BASE_CLR, HI_CLR]   # Tidak Hujan = lebih buruk → highlight

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(rain_pm25.index, rain_pm25.values,
                      color=colors_r, edgecolor='white', width=0.5)
        ax.set_ylabel('Rata-rata PM2.5 (µg/m³)')
        ax.set_title('Curah Hujan vs PM2.5\n(Gelap = kondisi lebih buruk)', fontweight='bold')
        for bar, val in zip(bars, rain_pm25.values):
            ax.text(bar.get_x() + bar.get_width()/2, val+1, f'{val:.0f}',
                    ha='center', fontsize=10, fontweight='bold')
        pct_r = (rain_pm25['Tidak Hujan'] - rain_pm25['Hujan']) / rain_pm25['Tidak Hujan'] * 100
        ax.annotate(f'Hujan turunkan\nPM2.5 {pct_r:.0f}%',
                    xy=(0, rain_pm25['Hujan']), xytext=(0.5, rain_pm25.mean()),
                    arrowprops=dict(arrowstyle='->', color='navy'), fontsize=8.5,
                    ha='center', color='navy')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.metric("Penurunan PM2.5 saat Hujan", f"{pct_r:.0f}%")

    # ── Korelasi ──────────────────────────────────────────────────────────────
    st.markdown("#### Matriks Korelasi: Faktor Meteorologi vs PM2.5")
    corr_cols = ['PM2.5','TEMP','PRES','DEWP','RAIN','WSPM']
    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={'size': 9}, linewidths=0.5)
    ax.set_title('Korelasi Antar Variabel', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("**Insight:** Angin kencang turunkan PM2.5 ~60%. Suhu dingin (<0°C) tingkatkan PM2.5 "
            "karena pemanas batu bara. Hujan efektif turunkan PM2.5 ~30–35%.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 – Geospasial
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Analisis Geospasial: Distribusi PM2.5 di Peta Beijing")
    st.markdown("Klik marker untuk detail stasiun. Ukuran lingkaran proporsional dengan tingkat polusi.")

    station_geo = df.groupby('station').agg(
        avg_pm25=('PM2.5', 'mean'),
        max_pm25=('PM2.5', 'max'),
        pct_unhealthy=('PM2.5', lambda x: (x > 55.4).mean() * 100)
    ).round(2)

    def get_color(pm25):
        if pm25 <= 35.4: return '#00e400'
        elif pm25 <= 55.4: return '#ffff00'
        elif pm25 <= 75:   return '#ff7e00'
        elif pm25 <= 100:  return '#ff0000'
        else: return '#8f3f97'

    m = folium.Map(location=[40.05, 116.37], zoom_start=10, tiles='CartoDB positron')
    for station, coords in STATION_COORDS.items():
        if station not in station_geo.index:
            continue
        avg = station_geo.loc[station, 'avg_pm25']
        max_v = station_geo.loc[station, 'max_pm25']
        pct = station_geo.loc[station, 'pct_unhealthy']
        folium.CircleMarker(
            location=coords, radius=max(8, avg / 7),
            color='white', weight=2, fill=True,
            fill_color=get_color(avg), fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>{station}</b><br>Rata-rata PM2.5: {avg:.1f} µg/m³"
                f"<br>Maks: {max_v:.0f} µg/m³<br>% Tidak Sehat: {pct:.1f}%",
                max_width=220),
            tooltip=f"{station}: {avg:.1f} µg/m³"
        ).add_to(m)

    st_folium(m, width=900, height=520)

    rank_df = station_geo.sort_values('avg_pm25', ascending=False).reset_index()
    rank_df.columns = ['Stasiun', 'Rata-rata PM2.5', 'Maks PM2.5', '% Tidak Sehat']
    rank_df.insert(0, 'Peringkat', range(1, len(rank_df)+1))
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    st.info("**Insight:** Stasiun pusat kota (Gucheng, Dongsi) paling terpolusi. "
            "Pinggiran kota (Dingling, Huairou) memiliki udara lebih bersih.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 – AQI Clustering
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Analisis Lanjutan: Clustering Kualitas Udara Berdasarkan AQI")
    st.markdown(
        "Setiap pengukuran diklasifikasikan ke kategori **AQI (Air Quality Index)** "
        "berdasarkan standar **US EPA** menggunakan metode **manual grouping (binning)**."
    )

    # pastikan AQI_Category tersedia
    if 'AQI_Category' not in df.columns or df['AQI_Category'].isna().all():
        df['AQI_Category'] = df['PM2.5'].apply(classify_aqi)

    available_cats = [c for c in AQI_ORDER if c in df['AQI_Category'].values]

    if not available_cats:
        st.warning("Tidak ada data AQI yang tersedia untuk filter yang dipilih.")
    else:
        c1, c2 = st.columns(2)

        # ── Distribusi Keseluruhan ────────────────────────────────────────────
        with c1:
            st.markdown("#### Distribusi Kategori AQI (Keseluruhan)")
            aqi_counts = df['AQI_Category'].value_counts()
            aqi_pct = pd.Series(
                {c: aqi_counts.get(c, 0) / len(df) * 100 for c in available_cats}
            ).round(2)
            short_labels = {
                'Good': 'Good', 'Moderate': 'Moderate',
                'Unhealthy for Sensitive Groups': 'USG',
                'Unhealthy': 'Unhealthy', 'Very Unhealthy': 'Very\nUnhealthy',
                'Hazardous': 'Hazardous'
            }

            fig, ax = plt.subplots(figsize=(7, 5))
            bar_colors = [AQI_COLORS.get(c, '#888') for c in aqi_pct.index]
            bars = ax.bar(range(len(aqi_pct)), aqi_pct.values,
                          color=bar_colors, edgecolor='white', width=0.75)
            ax.set_xticks(range(len(aqi_pct)))
            ax.set_xticklabels([short_labels.get(c, c) for c in aqi_pct.index], fontsize=9)
            ax.set_ylabel('Persentase Waktu (%)')
            ax.set_title('Distribusi Kategori AQI – Beijing 2013–2017', fontweight='bold')
            for bar, val in zip(bars, aqi_pct.values):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                        f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Distribusi per Stasiun ────────────────────────────────────────────
        with c2:
            st.markdown("#### Distribusi AQI per Stasiun")
            station_aqi_raw = (
                df.groupby(['station', 'AQI_Category'])
                .size()
                .unstack(fill_value=0)
            )
            existing_cols = [c for c in AQI_ORDER if c in station_aqi_raw.columns]
            if existing_cols:
                station_aqi_raw = station_aqi_raw[existing_cols]
                station_aqi_pct = station_aqi_raw.div(
                    station_aqi_raw.sum(axis=1), axis=0) * 100

                station_order = (df.groupby('station')['PM2.5'].mean()
                                 .sort_values(ascending=False).index.tolist())
                station_aqi_pct = station_aqi_pct.reindex(
                    [s for s in station_order if s in station_aqi_pct.index])

                fig, ax = plt.subplots(figsize=(7, 5))
                bottom = np.zeros(len(station_aqi_pct))
                for cat in station_aqi_pct.columns:
                    vals = station_aqi_pct[cat].fillna(0).values
                    ax.bar(range(len(station_aqi_pct)), vals, bottom=bottom,
                           color=AQI_COLORS.get(cat, '#888'), label=cat,
                           edgecolor='white', linewidth=0.5, width=0.85)
                    bottom += vals

                ax.set_xticks(range(len(station_aqi_pct)))
                ax.set_xticklabels(station_aqi_pct.index, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Persentase Waktu (%)')
                ax.set_title('Distribusi AQI per Stasiun\n(Tertinggi → Terendah PM2.5)',
                             fontweight='bold')
                ax.legend(title='AQI', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Data AQI per stasiun tidak tersedia.")

        # ── Tren AQI per Bulan ─────────────────────────────────────────────────
        st.markdown("#### Tren Distribusi AQI per Bulan")
        df_trend = df.copy()
        df_trend['month_year'] = (df_trend['datetime'].dt.to_period('M').astype(str))

        aqi_trend_raw = (
            df_trend.groupby(['month_year', 'AQI_Category'])
            .size()
            .unstack(fill_value=0)
        )
        trend_cols = [c for c in AQI_ORDER if c in aqi_trend_raw.columns]
        if trend_cols and len(aqi_trend_raw) > 0:
            aqi_trend_raw = aqi_trend_raw[trend_cols]
            row_sums = aqi_trend_raw.sum(axis=1)
            aqi_trend = aqi_trend_raw.div(row_sums.replace(0, 1), axis=0) * 100

            fig, ax = plt.subplots(figsize=(14, 5))
            bottom = np.zeros(len(aqi_trend))
            for cat in aqi_trend.columns:
                vals = aqi_trend[cat].fillna(0).values
                ax.bar(range(len(aqi_trend)), vals, bottom=bottom,
                       color=AQI_COLORS.get(cat, '#888'), label=cat, width=1.0)
                bottom += vals

            step = max(1, len(aqi_trend) // 10)
            tick_pos = list(range(0, len(aqi_trend), step))
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([aqi_trend.index[i] for i in tick_pos],
                               rotation=30, ha='right', fontsize=8)
            ax.set_ylabel('Persentase (%)')
            ax.set_title('Tren Distribusi Kategori AQI per Bulan', fontweight='bold')
            ax.legend(title='AQI', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Data tren AQI tidak cukup untuk ditampilkan.")

        # ── Ringkasan ──────────────────────────────────────────────────────────
        st.markdown("#### Ringkasan % Waktu per Kategori AQI per Stasiun")
        if 'station_aqi_pct' in dir() and existing_cols:
            try:
                summary_df = station_aqi_pct.round(1)
                st.dataframe(
                    summary_df.style.background_gradient(cmap='RdYlGn_r', axis=None),
                    use_container_width=True
                )
            except Exception:
                st.dataframe(station_aqi_pct.round(1), use_container_width=True)

    st.info("**Insight:** Hanya ~7% waktu kualitas udara Beijing berada di kategori 'Good'. "
            "Lebih dari 50% waktu berada di kategori 'Unhealthy' atau lebih buruk.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:12px'>"
    "Proyek Analisis Data | Alfath Septyan | "
    "Sumber: PRSA Air Quality Dataset Beijing 2013–2017</div>",
    unsafe_allow_html=True
)
