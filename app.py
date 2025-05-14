import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, anderson, norm, zscore, probplot

st.title("📊 Normallik Testleri Uygulaması")

# Dosya yükleme
uploaded_file = st.file_uploader("Bir Excel dosyası yükleyin", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("Excel dosyası başarıyla yüklendi.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox("Normallik testi yapılacak sütunu seçin", numeric_cols)

        subset_option = st.checkbox("Başka bir sütuna göre gruplu test yapmak ister misiniz?")

        if subset_option:
            group_col = st.selectbox("Gruplamak istediğiniz sütunu seçin", df.columns)
            groups = df[group_col].dropna().unique()
            selected_groups = st.multiselect("Test edilecek grupları seçin", groups)

            if selected_groups:
                for group in selected_groups:
                    st.markdown(f"### Grup: {group}")
                    data = df[df[group_col] == group][selected_col].dropna()

                    if len(data) < 3:
                        st.warning(f"{group} grubunda yeterli veri yok (min 3 gözlem).")
                        continue

                    # Shapiro-Wilk
                    shapiro_stat, shapiro_p = shapiro(data)

                    # Kolmogorov-Smirnov (z-score normalize)
                    z_data = zscore(data.dropna())
                    ks_stat, ks_p = kstest(z_data, 'norm')

                    # Anderson-Darling
                    ad_result = anderson(data, dist='norm')

                    # Sonuçları göster
                    st.write("**Shapiro-Wilk Testi**")
                    st.write(f"Test istatistiği: {shapiro_stat:.4f}, p-değeri: {shapiro_p:.4f}")

                    if shapiro_p > 0.05:
                        st.write("Not: p-değeri 0.05'ten büyük, normal dağılıma uygun.")
                    else:
                        st.write("Not: p-değeri 0.05'ten küçük, normal dağılıma uygun değil.")

                    st.write("**Kolmogorov-Smirnov Testi (z-score normalize)**")
                    st.write(f"Test istatistiği: {ks_stat:.4f}, p-değeri: {ks_p:.4f}")

                    if ks_p > 0.05:
                        st.write("Not: p-değeri 0.05'ten büyük, normal dağılıma uygun.")
                    else:
                        st.write("Not: p-değeri 0.05'ten küçük, normal dağılıma uygun değil.")

                    st.write("**Anderson-Darling Testi**")
                    st.write(f"Test istatistiği: {ad_result.statistic:.4f}")
                    for i in range(len(ad_result.critical_values)):
                        st.write(f"Kritik değer (%{ad_result.significance_level[i]}): {ad_result.critical_values[i]:.4f}")

                    # Histogram
                    fig_hist = plt.figure(figsize=(6, 4))
                    sns.histplot(data, kde=True)
                    plt.title('Histogram')
                    st.pyplot(fig_hist)

                    # Q-Q Plot
                    fig_qq = plt.figure(figsize=(6, 4))
                    probplot(data, dist="norm", plot=plt)
                    plt.title('Q-Q Plot')
                    st.pyplot(fig_qq)

        else:
            data = df[selected_col].dropna()

            if len(data) >= 3:
                # Shapiro-Wilk
                shapiro_stat, shapiro_p = shapiro(data)

                # Kolmogorov-Smirnov (z-score normalize)
                z_data = zscore(data.dropna())
                ks_stat, ks_p = kstest(z_data, 'norm')

                # Anderson-Darling
                ad_result = anderson(data, dist='norm')

                st.subheader("🔍 Normallik Testi Sonuçları")

                st.write("**Shapiro-Wilk Testi**")
                st.write(f"Test istatistiği: {shapiro_stat:.4f}, p-değeri: {shapiro_p:.4f}")

                if shapiro_p > 0.05:
                    st.write("Not: p-değeri 0.05'ten büyük, normal dağılıma uygun.")
                else:
                    st.write("Not: p-değeri 0.05'ten küçük, normal dağılıma uygun değil.")

                st.write("**Kolmogorov-Smirnov Testi (z-score normalize)**")
                st.write(f"Test istatistiği: {ks_stat:.4f}, p-değeri: {ks_p:.4f}")

                if ks_p > 0.05:
                    st.write("Not: p-değeri 0.05'ten büyük, normal dağılıma uygun.")
                else:
                    st.write("Not: p-değeri 0.05'ten küçük, normal dağılıma uygun değil.")

                st.write("**Anderson-Darling Testi**")
                st.write(f"Test istatistiği: {ad_result.statistic:.4f}")
                for i in range(len(ad_result.critical_values)):
                    st.write(f"Kritik değer (%{ad_result.significance_level[i]}): {ad_result.critical_values[i]:.4f}")

                # Histogram
                fig_hist = plt.figure(figsize=(6, 4))
                sns.histplot(data, kde=True)
                plt.title('Histogram')
                st.pyplot(fig_hist)

                # Q-Q Plot
                fig_qq = plt.figure(figsize=(6, 4))
                probplot(data, dist="norm", plot=plt)
                plt.title('Q-Q Plot')
                st.pyplot(fig_qq)

            else:
                st.warning("Seçilen sütun için yeterli veri yok (min 3 gözlem).")

    else:
        st.warning("Excel dosyasında sayısal sütun bulunamadı.")
