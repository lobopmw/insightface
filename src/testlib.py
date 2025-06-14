import face_recognition as fr

# Carregar uma imagem de teste
image = fr.load_image_file("images/images_students/Starley do Nascimento Lobo/2.jpg")

# Localizar rostos na imagem
face_locations = fr.face_locations(image)

# Checar se encontrou rostos
if not face_locations:
    print("Nenhum rosto detectado")
else:
    # Obter os encodings dos rostos
    face_encodings = fr.face_encodings(image, known_face_locations=face_locations)

    for encoding in face_encodings:
        print("Rosto detectado e encoding gerado com sucesso")
