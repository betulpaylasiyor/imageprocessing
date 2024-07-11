import streamlit as st  # Streamlit modülünü içe aktarma
import cv2  # OpenCV modülünü içe aktarma
import numpy as np  # NumPy modülünü içe aktarma
from PIL import Image  # Pillow modülünden Image sınıfını içe aktarma

# Görüntüyü yükleme fonksiyonu
def load_image(image_file):
    img = Image.open(image_file)  # Görüntü dosyasını açma
    return img  # Görüntüyü döndürme

# Histogram Hesaplama fonksiyonu
def calculate_histogram(image):
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # Görüntüyü gri tonlamaya çevirme
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])  # Histogramı hesaplama
    return hist  # Histogramı döndürme

# Histogram Eşitleme fonksiyonu
def equalize_histogram(image):
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # Görüntüyü gri tonlamaya çevirme
    equalized_img = cv2.equalizeHist(img_gray)  # Histogramı eşitleme
    return equalized_img  # Eşitlenmiş görüntüyü döndürme

# Kenar Çıkarma fonksiyonu
def edge_detection(image):
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # Görüntüyü gri tonlamaya çevirme
    edges = cv2.Canny(img_gray, 100, 200)  # Kenar çıkarma işlemi
    return edges  # Kenarları döndürme

# Bulanıklaştırma fonksiyonu
def blur_image(image):
    img_array = np.array(image)  # Görüntüyü NumPy dizisine çevirme
    blurred_img = cv2.GaussianBlur(img_array, (5, 5), 0)  # Görüntüyü bulanıklaştırma
    return blurred_img  # Bulanık görüntüyü döndürme

# Keskinleştirme fonksiyonu
def sharpen_image(image):
    img_array = np.array(image)  # Görüntüyü NumPy dizisine çevirme
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Keskinleştirme çekirdeği tanımlama
    sharpened_img = cv2.filter2D(img_array, -1, kernel)  # Görüntüyü keskinleştirme
    return sharpened_img  # Keskinleştirilmiş görüntüyü döndürme

# Region Growing Segmentasyon fonksiyonu
def region_growing(image, seed_point):
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # Görüntüyü gri tonlamaya çevirme
    segmented = np.zeros_like(img_gray)  # Boş bir segmentasyon görüntüsü oluşturma
    segmented[seed_point[1], seed_point[0]] = 255  # Tohum noktasını beyaz yapma
    queue = [seed_point]  # Kuyruğa tohum noktasını ekleme
    while queue:  # Kuyruk boş olana kadar
        x, y = queue.pop(0)  # Kuyruğun başından elemanı çıkarma
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Komşu pikseller için döngü
            nx, ny = x + dx, y + dy  # Yeni komşu piksel koordinatları
            if 0 <= nx < img_gray.shape[1] and 0 <= ny < img_gray.shape[0] and segmented[ny, nx] == 0 and img_gray[ny, nx] < 128:
                segmented[ny, nx] = 255  # Komşu pikseli beyaz yapma
                queue.append((nx, ny))  # Kuyruğa yeni komşu pikseli ekleme
    return segmented  # Segmentlenmiş görüntüyü döndürme

# Streamlit Arayüzü
st.title("Görüntü İşleme Uygulaması")  # Uygulama başlığı
st.write("Lütfen bir görüntü yükleyin ve işlemek istediğiniz işlemi seçin.")  # Açıklama metni

image_file = st.file_uploader("Görüntü Yükle", type=["jpg", "png", "jpeg"])  # Görüntü yükleme bileşeni

if image_file is not None:  # Eğer bir görüntü yüklendiyse
    img = load_image(image_file)  # Görüntüyü yükle
    st.image(img, caption="Yüklenen Görüntü", use_column_width=True)  # Görüntüyü göster
    
    option = st.selectbox("İşlem Seçin", ["Histogram Hesapla", "Histogram Eşitle", "Kenar Çıkarma", "Bulanıklaştırma", "Keskinleştirme", "Region Growing Segmentasyon"])  # İşlem seçimi

    if option == "Histogram Hesapla":  # Histogram hesaplama seçildiyse
        hist = calculate_histogram(img)  # Histogramı hesapla
        st.bar_chart(hist)  # Histogramı göster

    elif option == "Histogram Eşitle":  # Histogram eşitleme seçildiyse
        equalized_img = equalize_histogram(img)  # Histogramı eşitle
        st.image(equalized_img, caption="Histogram Eşitlenmiş Görüntü", use_column_width=True)  # Eşitlenmiş görüntüyü göster

    elif option == "Kenar Çıkarma":  # Kenar çıkarma seçildiyse
        edges = edge_detection(img)  # Kenarları çıkar
        st.image(edges, caption="Kenarlar", use_column_width=True)  # Kenarları göster

    elif option == "Bulanıklaştırma":  # Bulanıklaştırma seçildiyse
        blurred_img = blur_image(img)  # Görüntüyü bulanıklaştır
        st.image(blurred_img, caption="Bulanık Görüntü", use_column_width=True)  # Bulanık görüntüyü göster

    elif option == "Keskinleştirme":  # Keskinleştirme seçildiyse
        sharpened_img = sharpen_image(img)  # Görüntüyü keskinleştir
        st.image(sharpened_img, caption="Keskin Görüntü", use_column_width=True)  # Keskin görüntüyü göster

    elif option == "Region Growing Segmentasyon":  # Region growing segmentasyon seçildiyse
        seed_x = st.number_input("Tohum Noktası X Koordinatı", min_value=0, max_value=img.size[0], value=img.size[0]//2)  # Tohum noktası X koordinatı
        seed_y = st.number_input("Tohum Noktası Y Koordinatı", min_value=0, max_value=img.size[1], value=img.size[1]//2)  # Tohum noktası Y koordinatı
        segmented_img = region_growing(img, (seed_x, seed_y))  # Region growing segmentasyon işlemini yap
        st.image(segmented_img, caption="Segmentlenmiş Görüntü", use_column_width=True)  # Segmentlenmiş görüntüyü göster

