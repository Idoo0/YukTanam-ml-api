from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from components.parser import getParserComponent
from components.llm import getLLM
from others.api.plantName import getPlantName
import numpy as np
import base64

BASE_MODEL_PATH = "./models/"

class Predictor:

    def __init__(self):
        self._prepareChain()
        self._prepareModel()
        self.class_names = [
            'Bacterial_spot', 'Black_Rot', 'Black_Spot', 'Brown_Leaf_Tips',
            'Downy_Mildew', 'Early_Blight', 'Healthy', 'Late_Blight', 'Leaf_Scorch',
            'Leaf_Spot', 'Powdery_Mildew', 'Rust', 'Spider_Mite', 'Wilted_plant', 'Yellow_leaves'
        ]

    def _prepareChain(self):
        print(f"loading chain comps...")
        try:
            self.parserComp = getParserComponent()
            self.llm = getLLM()
        except Exception as e:
            print(f"Error loading chain comps : {e}")
            return "load error"
        print("chain loaded!")
        
    def _prepareModel(self):
        names = [
            "resnet50_aug.h5"
            # "vitMobile.h5"
        ]
        self.models = {}
        for name in names:
            print(f"loading model {name}...")
            model_name = name.split('.')[0]
            try:
                self.models[model_name] = load_model(BASE_MODEL_PATH + name)
            except Exception as e:
                print(f"Error loading model : {e}")
                return "load error"
        print("model loaded!")

    def _preprocessImage(self, image_path, image_height, image_widht):
        img = load_img(image_path, target_size=(image_height, image_widht))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, 0)
        return img_array

    def _predictImage(self, image_array, model, class_names):
        predictions = model.predict(image_array)
        # predicted_class = class_names[np.argmax(predictions)]
        # confidence = np.max(predictions)
        details = "\n".join(
            [
                f"{class_name}: {probability:.4f}"
                for class_name, probability in sorted(
                    zip(class_names, predictions[0]),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
        )
        return details

    def _getWrappedResponse(self, disease_predictions, pn_predictions, image_data):
        system_prompt = """
            Kamu adalah seorang ahli penyakit tumbuhan. Tugasmu adalah membantu mendeteksi jenis tanaman dan penyakit yang menyerang berdasarkan hasil prediksi dari model. Informasi yang diberikan oleh pengguna meliputi gambar tanaman dan hasil prediksi model berupa nama tanaman dan jenis penyakitnya.\n\nTugasmu adalah:\n\nMengidentifikasi Nama Tanaman berdasarkan hasil prediksi model.\nMemberikan informasi tentang Jenis Penyakit berdasarkan prediksi.\nMenjelaskan Deskripsi Penyakit, termasuk penyebab dan dampaknya pada tanaman.\nMemberikan saran lengkap tentang Cara Penanganan, termasuk metode pencegahan dan langkah-langkah perawatan.\nFormat output yang dihasilkan harus seperti berikut:\n\nNama Tanaman: [Nama tanaman berdasarkan prediksi]\n\nJenis Penyakit: [Nama penyakit berdasarkan prediksi]\n\nDeskripsi Penyakit: [Penjelasan singkat tentang penyakit, penyebab, gejala, dan dampaknya]\n\nCara Penanganan: [Langkah-langkah perawatan dan pencegahan untuk penyakit ini]\n\nJika informasi dari model tidak cukup atau tidak jelas, gunakan pengetahuan yang relevan untuk melengkapi penjelasan. Jawablah secara profesional dan mudah dipahami. Nantinya kamu akan menerima gambar, dengan hasil prediksi dan pengetahuan mu tentukan penyakitnya,
            anda juga berhak menentukan point yang didapatkan 0-30 Penyakit sangat parah, 30-59 Penyakit tidak parah, 60-100 Tanaman sehat. tentuakan berdasarkan kondisi tanaman. pilih angka eksak
            Ingatlah hanya untuk mengembalikan hal yang saya minta, jangan menambah hal lain dari 5 poin itu. anda adalah ahli oleh karenanya JANGAN MENAMBAHKAN CATATAN ATAU CATATAN PENTING saya mohon untuk memastikan hanya 4 section itu yang kamu berikan yaitu (nama tanaman, jenis penyakit, deskripsi penyakit, dan cara penanggulangan),
            Pada bagian nama tanaman langsungkan saja, tulis yang kamu ketahui, ingat hanya nama tanaman saja jangan tambahkan apapun, jangan tambahkan kata-kata asumsi atau apapun itu
        """
        
        if pn_predictions is None:
            pn_data = ''
        else:
            pn_data = f"""
            Gunakan ini untuk mendapatkan nama tanaman, sesuaikan lagi dengan pengetahuan dan penglihatan anda
            {pn_predictions}
            """
        hm = HumanMessage(
            content=[
                {"type": "text", "text": f"the model prediction is {disease_predictions} {pn_data}, jawablah dalam bentuk {self.parserComp["format_instructions"]}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )
        
        message = ChatPromptTemplate([
            ("system", system_prompt),
            hm
        ])
        chain = message | self.llm | self.parserComp["parser"]
        result = chain.invoke({})

        return result
    
    def _predict(self, image_path, model, image_widht, image_height):
        
        # preprocess the image!
        image = self._preprocessImage(
            image_path=image_path,
            image_height=image_height,
            image_widht=image_widht
        )
        
        # predict the image!
        predictions = self._predictImage(
            image_array=image,
            model=model,
            class_names=self.class_names
        )
        
        # (OPT) predict the name!
        plant_name = getPlantName(image_path=image_path)
        
        # wrapped the result!
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
        result = self._getWrappedResponse(
            disease_predictions=predictions,
            pn_predictions=plant_name,
            image_data=image_data
        )
        return result
    
    def predictResnetV1(self, image_path):

        result = self._predict(
            image_path=image_path,
            model=self.models["resnet50_aug"],
            image_height=224,
            image_widht=224
        )

        return result