Deteksi Jenis Sampah (YOLO11)

Aplikasi web untuk mendeteksi sampah organik, anorganik, residu, dan B3 dari gambar atau kamera langsung. Model YOLO11n diekspor ke ONNX dan dijalankan dengan ONNX Runtime (tanpa PyTorch), sehingga muat di hosting RAM 512 MB.

Demo: https://deteksi-sampah-b859f.containers.snapdeploy.app (server tertidur setelah 15 menit tidak dipakai; akses pertama bisa memakan waktu sekitar satu menit)

Struktur
Deploy/
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── main.py            # API + penyaji frontend
├── index.html         # tampilan web
├── README.MD 
└── models/
    ├── best.onnx      # model utama (input 640x640)
    └── best_416.onnx  # opsional: model cepat untuk kamera (input 416x416)

Endpoint
Method |        Path	   |                        Fungsi                              |
GET    |  /	               | Halaman web                                                |
GET	   | /health	       | Status server, model, dan memori                           |
POST   | /v1/predict?      | Deteksi (form-data, field file)                            |
       | conf=0.347&fast=0 |                                                            |
GET	   | /v1/stats	       | Jumlah request, error rate, memori, latensi p50/p95/p99    |
GET	   | /docs	           | Dokumentasi interaktif                                     |

Setiap respons membawa header X-Request-ID, dan satu baris log JSON dicetak per request.

Environment variable (opsional)
     Variabel	  |          Default	  |                 Keterangan                                                  |
PORT	          | 1024	              | Port server                                                                 |
MODEL_PATH	      | models/best.onnx	  | Model utama                                                                 |
FAST_MODEL_PATH   | models/best_416.onnx  | Model cepat (dipakai jika file ada)                                         |
CONF_THRESHOLD    | 0.347	              | Ambang confidence bawaan                                                    |
ORT_THREADS       | 1	                  | Jumlah thread ONNX Runtime; naikkan jika hosting punya lebih dari 1 vCPU    |
MAX_SIDE	      | 1600	              | Sisi terpanjang gambar yang diproses (px)                                   |

Jalankan lokal
cd Deploy
pip install -r requirements.txt
uvicorn main:app --port 8000

Jalankan dengan Docker
cd Deploy
docker build -t deteksi-sampah .
docker run --rm -p 1024:1024 deteksi-sampah

Ekspor model
yolo export model=runs/sampah_yolo11n-4/weights/best.pt format=onnx imgsz=640 simplify=True
yolo export model=runs/sampah_yolo11n-4/weights/best.pt format=onnx imgsz=416 simplify=True

Perintah ekspor menghasilkan best.onnx di folder yang sama, jadi ganti namanya (misalnya best_416.onnx) sebelum mengekspor ukuran berikutnya.