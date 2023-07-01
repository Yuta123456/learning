from PIL import Image, ImageDraw, ImageFont

def save_combined_image(src_img_path, image_paths, output_path):
    images = []

    # 元画像を読み込んでリストに追加する
    src_image = Image.open(src_img_path)
    images.append(src_image)

    # 画像を読み込んでリストに追加する
    for path in image_paths:
        image = Image.open(path)
        images.append(image)

    # 画像の幅と高さを取得
    width, height = images[0].size

    # 隙間として使用するピクセル数を定義
    gap = 5

    # 結合された画像の幅と高さを計算
    combined_width = width * 5 + gap * 4
    combined_height = height * 3 + gap
    # 元画像と「入力」という文字列を結合した画像を作成
    input_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    input_image.paste(images[0], (0, 0))
    input_text = "input"
    text_color = (0, 0, 0)  # テキストの色を黒に設定
    text_font = ImageFont.truetype("arial.ttf", 16, encoding='utf-8')  # テキストのフォントとサイズを設定
    draw = ImageDraw.Draw(input_image)
    # text_width, text_height = draw.textsize(input_text, font=text_font)
    text_x = 0 # 右端から5ピクセルの余白を開ける
    text_y = 0  # 元画像の上部中央に配置する
    draw.text((text_x, text_y), input_text, font=text_font, fill=text_color)
    
    combined_width = width * 5 + gap * 4
    combined_height = height * 2 + gap
    # 結合された画像のキャンバスを作成
    combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))

    # 画像を結合して保存する
    for i, image in enumerate(images[1:11]):  # 元画像以外の画像を結合する
        x = (i % 5) * (width + gap)
        y = (i // 5) * (height + gap)
        combined_image.paste(image, (x, y))

    # 元画像と結合された画像を結合する
    input_image.paste(combined_image, (0, height + gap))

    input_image.save(output_path)