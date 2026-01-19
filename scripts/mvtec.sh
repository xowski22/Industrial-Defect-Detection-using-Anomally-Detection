URL="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"
DIR="data/raw"

mkdir -p $DIR
cd $DIR

echo "Downloading MVTec Anomaly Detection dataset..."
wget -O mvtec_anomaly_detection.tar.xz $URL

mkdir -p mvtec_anomaly_detection

echo "Extracting dataset..."
tar -xf mvtec_anomaly_detection.tar.xz -C mvtec_anomaly_detection --strip-components=0

rm mvtec_anomaly_detection.tar.xz