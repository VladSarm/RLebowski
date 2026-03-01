import numpy as np
from PIL import Image


# Цвета боулера в оригинальной игре (для детекта)
_BOWLER_COLORS = [
    (198, 89, 179),   # розовая голова/шар
    (84, 92, 214),    # тело (голубой)
    (66, 72, 200),    # ноги (синий)
    (45, 50, 184),    # рука/тёмно-синий
]


def _is_bowler_pixel(r, g, b):
    for bc in _BOWLER_COLORS:
        if abs(int(r) - bc[0]) < 15 and abs(int(g) - bc[1]) < 15 and abs(int(b) - bc[2]) < 15:
            return True
    return False


def find_bowler(frame):
    """Найти bounding box боулера. Фильтруем линии дорожки (>15px в ряд)."""
    h, w = frame.shape[:2]
    min_x, min_y = w, h
    max_x, max_y = 0, 0
    found = False
    for y in range(h // 2, h - 30):
        row_count = 0
        for x in range(0, w // 3):
            if _is_bowler_pixel(frame[y, x, 0], frame[y, x, 1], frame[y, x, 2]):
                row_count += 1
        if row_count > 15:
            continue
        for x in range(0, w // 3):
            if _is_bowler_pixel(frame[y, x, 0], frame[y, x, 1], frame[y, x, 2]):
                if min_x > x: min_x = x
                if min_y > y: min_y = y
                if max_x < x: max_x = x
                if max_y < y: max_y = y
                found = True
    if found:
        return min_x, min_y, max_x, max_y
    return None


def draw_lebowski_game():
    """
    Игровой спрайт Dude (14x36) — полностью перекрывает боулера.
    Вид сбоку, смотрит вправо. Халат, белая майка, очки, борода, длинные волосы.
    """
    W, H = 14, 36
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    px = img.load()

    SKIN     = (222, 178, 133, 255)
    HAIR     = (160, 130, 80, 255)
    HAIR_DK  = (130, 100, 60, 255)
    BEARD    = (120, 95, 60, 255)
    GLASSES  = (15, 15, 15, 255)
    ROBE     = (200, 185, 160, 255)
    ROBE_SH  = (175, 160, 135, 255)
    SHIRT    = (240, 240, 235, 255)
    SHORTS   = (90, 85, 75, 255)
    SANDAL   = (140, 100, 60, 255)

    def p(x, y, c):
        if 0 <= x < W and 0 <= y < H:
            px[x, y] = c

    def rect(x1, y1, x2, y2, c):
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                p(x, y, c)

    # -- Волосы (макушка, густые) --
    rect(4, 0, 9, 0, HAIR)
    rect(3, 1, 10, 1, HAIR)
    rect(3, 2, 10, 2, HAIR)
    rect(3, 3, 10, 3, HAIR)

    # -- Волосы назад (затылок, длинные до плеч) --
    rect(2, 3, 4, 10, HAIR)
    rect(1, 5, 3, 12, HAIR_DK)

    # -- Чёлка спереди --
    rect(9, 1, 11, 3, HAIR)

    # -- Лицо (профиль) --
    rect(5, 4, 9, 4, SKIN)
    rect(5, 5, 10, 6, SKIN)
    rect(5, 7, 9, 8, SKIN)

    # -- Нос --
    p(11, 5, SKIN)
    p(11, 6, SKIN)

    # -- Очки --
    rect(9, 4, 11, 5, GLASSES)
    rect(9, 6, 10, 6, GLASSES)

    # -- Борода --
    rect(7, 9, 10, 9, BEARD)
    rect(7, 10, 9, 10, BEARD)

    # -- Шея --
    rect(5, 9, 7, 10, SKIN)

    # -- Халат + майка (торс) --
    rect(3, 11, 10, 11, ROBE)      # плечи
    rect(3, 12, 4, 20, ROBE)       # спина халата
    rect(9, 12, 11, 20, ROBE)      # перед халата
    rect(5, 12, 8, 14, SHIRT)      # майка видна
    rect(5, 15, 9, 17, SHIRT)      # пузико!
    rect(6, 16, 10, 17, SHIRT)     # пузико выпирает
    rect(4, 19, 10, 20, ROBE_SH)   # пояс халата

    # -- Рука передняя --
    rect(11, 13, 12, 15, ROBE)     # рукав
    rect(11, 16, 13, 18, SKIN)     # предплечье

    # -- Рука задняя --
    rect(1, 13, 3, 15, ROBE)
    rect(0, 16, 2, 18, SKIN)

    # -- Шорты --
    rect(4, 21, 10, 25, SHORTS)
    rect(4, 26, 6, 27, SHORTS)     # левая штанина
    rect(8, 26, 10, 27, SHORTS)    # правая штанина

    # -- Ноги --
    rect(4, 28, 6, 30, SKIN)
    rect(8, 28, 10, 30, SKIN)

    # -- Шлёпанцы --
    rect(3, 31, 7, 32, SANDAL)
    rect(7, 31, 11, 32, SANDAL)

    # -- Подошва --
    rect(3, 33, 7, 35, SANDAL)
    rect(7, 33, 11, 35, SANDAL)

    return np.array(img)


def replace_bowler(frame, dude_sprite):
    """Найти боулера и нарисовать Dude поверх него."""
    result = frame.copy()
    bbox = find_bowler(frame)
    if bbox is None:
        return result

    bx1, _, bx2, by2 = bbox
    dude_h, dude_w = dude_sprite.shape[:2]
    bowler_cx = (bx1 + bx2) // 2
    dx = bowler_cx - dude_w // 2
    dy = (by2 + 1) - dude_h

    for sy in range(dude_h):
        for sx in range(dude_w):
            a = dude_sprite[sy, sx, 3]
            if a == 0:
                continue
            tx, ty = dx + sx, dy + sy
            if 0 <= tx < result.shape[1] and 0 <= ty < result.shape[0]:
                if a == 255:
                    result[ty, tx] = dude_sprite[sy, sx, :3]
                else:
                    alpha = a / 255.0
                    result[ty, tx] = (
                        dude_sprite[sy, sx, :3].astype(float) * alpha +
                        result[ty, tx].astype(float) * (1 - alpha)
                    ).astype(np.uint8)

    return result
