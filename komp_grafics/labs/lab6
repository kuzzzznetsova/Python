# Лабораторная работа №6
# Тема: Создание шейдерных анимационных эффектов в OpenGL
# Вариант № 10 : Прямой цилиндр, основание – сектор гиперболы
# Освещение: Зеркальное освещение от источника света, расположенного в заданной позиции
# Кузнецова Дарина
# группа: М8О-305Б-20
from math import *
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def vec_length(vec):
    return sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

def vec_norm(vec):
    length = vec_length(vec)
    if length == 0:
        return (0, 0, 0)
    norm = (vec[0] / length, vec[1] / length, vec[2] / length)
    return norm


def cross_product(v1, v2):
    return (v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0])


def dot_product(v1, v2):
    ret = 0.0
    for i in range(len(v1)):
        ret += v1[i] * v2[i]
    return ret


angle = pi / 1960
SCREEN_SIZE = (1024, 800)
vectors = [(10, 32), (20, 47)]
# WHITE = (255, 255, 255)
BLACK = (200, 200, 0)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)
color = (0.0, 0.0, 205, 1)
# LIGHT_GRAY = (220, 220, 220)

screen_vect1 = (1, 0, 0)
screen_vect2 = (0, 1, 0)
screen_indices = vec_norm(cross_product(screen_vect1, screen_vect2))
show_indices = True
show_normals = True
scale = 1.0
centr = (SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2)
note = 5

def down(angle):
    return (cos(angle), sin(angle, 0))

def up(angle):
    return (cos(angle), sin(angle, 2))


def hyperbola_2d(n, z):
    l = -1
    r = 1
    points = []
    for i in range(l * (n - 1), r * (n - 1) + 1, r - l):
        x = i / (n - 1)
        y = 2/(2*x+3)
        points.append((x, y, z))
    return points


def hyperbola_3d(n, h):
    vertices = []
    vertices.extend(hyperbola_2d(n, h / 2))
    vertices.extend(hyperbola_2d(n, -h / 2))
    faces = []
    face1 = []
    for i in range(n):
        face1.append(i)
    faces.append(list(reversed(face1)))
    face2 = []
    for j in range(n, 2 * n):
        face2.append(j)
    faces.append(face2)
    for j in range(n):
        if j == (n - 1):
            faces += [[j, 0, n, 2 * n - 1]]
        else:
            faces += [[j, j + 1, j + n + 1, j + n]]
    return vertices, faces

def cube():
    coords = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    faces = [(0, 1, 2, 3), (1, 5, 6, 2), (2, 6, 7, 3), (5, 4, 7, 6), (0, 3, 7, 4), (0, 4, 5, 1)]
    return coords, faces

models = [(*hyperbola_3d(5, 1), "Hyperbola"), (*cube(), "Cube")]
current_model = 0
vertices, faces, name = models[current_model]

def next_model(direction):
    global vertices, faces, current_model, name
    current_model = (current_model + direction) % len(models)
    vertices, faces, name = models[current_model]

def normal(face):
    vertices = models[current_model][0]
    ver1 = vertices[face[0]]
    ver2 = vertices[face[1]]
    ver3 = vertices[face[2]]
    vec1 = (ver2[0] - ver1[0], ver2[1] - ver1[1], ver2[2] - ver1[2])
    vec2 = (ver3[0] - ver2[0], ver3[1] - ver2[1], ver3[2] - ver2[2])
    return vec_norm(cross_product(vec1, vec2))

def visible_face(face):
    norm = normal(face)
    if dot_product(screen_indices, norm) <= 0:
        return True
    else:
        return False


def filter_faces():
    visible = []
    invisible = []
    for face in faces:
        if visible_face(face):
            visible += [face]
        else:
            invisible += [face]
    return visible, invisible


def rotate_x(angle):
    MX = [[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]]
    return MX

def rotate_y(angle):
    MY = [[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]]
    return MY

def rotate_z(angle):
    MZ = [[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]]
    return MZ

def rotate(angle, rot):
    m = rot(angle)
    for i in range(len(vertices)):
        vertices[i] = product(m, vertices[i])


def product(m, v):
    result = []
    for row in m:
        ret = 0
        for i in range(0, len(row)):
            xm = row[i]
            xv = v[i]
            ret += xv * xm
        result.append(ret)
    return result


def to_scr(pos):
    x = pos[0] * scale + centr[0]
    y = pos[1] * scale + centr[1]
    return (x, y)


def orthogonal(pos):
    return (pos[0], pos[1])


def draw_wireframe_face(screen, color, face):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, pointdata)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    for i in range(len(face)):
        a = vertices[face[i]]
        if (i + 1) == len(face):
            b = vertices[face[0]]
        else:
            b = vertices[face[i + 1]]
        ortho_a = orthogonal(a)
        screen_ortho_a = to_scr(ortho_a)
        ortho_b = orthogonal(b)
        screen_ortho_b = to_scr(ortho_b)
        pygame.draw.line(screen, color, screen_ortho_a, screen_ortho_b, 5)


def draw_shaded_face(screen, color, face):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, pointdata)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    points = []
    for i in range(len(face)):
        points.append(to_scr(orthogonal(vertices[face[i]])))
    very_new_color = turn_to_color(color, face)
    pygame.draw.polygon(screen, very_new_color, points)
    return


def draw_wire_GL():
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, pointdata)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    mode = GL_POLYGON
    coords, faces, _ = models[current_model]
    glMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE, color)
    glColor3d(0, 1, 0)
    for face in faces[:]:
        glBegin(mode)
        for idx in reversed(face):
            glVertex3fv(coords[idx])
        glEnd()

def turn_to_color(base_color, face):
    face_normal = normal(face)
    dot_face_normal = -dot_product(face_normal, screen_indices)
    new_color = (weighted_average(base_color[0] / 2, base_color[0], dot_face_normal),
                 weighted_average(base_color[1] / 2, base_color[1], dot_face_normal),
                 weighted_average(base_color[2] / 2, base_color[2], dot_face_normal))
    return new_color


def weighted_average(left, right, alpha):
    return int(alpha * left + (1 - alpha) * right)


def draw_shaded(screen, color, faces):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, pointdata)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    for face in faces:
        draw_shaded_face(screen, color, face)


def draw_wireframe(screen, color, faces):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, pointdata)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    for face in faces:
        draw_wireframe_face(screen, color, face)


def draw_vec_center(screen, color, vec):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, pointdata)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    x = vec[0]
    y = vec[1]
    pygame.draw.line(screen, color, centr, (SCREEN_SIZE[0] // 2 + x, SCREEN_SIZE[1] // 2 - y), 3)
    return


def draw_vert_index(screen, font):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, pointdata)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    if not show_indices:
        return
    for i in range(len(vertices)):
        x = to_scr(orthogonal(vertices[i]))
        pic = font.render("%d" % i, False, BLACK)
        screen.blit(pic, x)


def draw_normals(screen):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, pointdata)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    if not show_normals:
        return
    for face in faces:
        norm = normal(face)
        mid = vertices[face[0]]
        norm = (mid[0] + norm[0], mid[1] + norm[1], mid[2] + norm[2])
        pygame.draw.line(screen, BLACK, to_scr(mid), to_scr(norm), 2)
    return




def create_shader(shader_type, source):
    # Создаем пустой объект шейдера
    shader = glCreateShader(shader_type)
    # Привязываем текст шейдера к пустому объекту шейдера
    glShaderSource(shader, source)
    # Компилируем шейдер
    glCompileShader(shader)
    # Возвращаем созданный шейдер
    glVertexPointer(3, GL_FLOAT, 0, pointdata)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    return shader

def main():
    global scale, note, show_indices, show_normals, pointdata, pointcolor
    pointcolor = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    # Запускаем основной цикл
    pointdata = vertices
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((1024, 800), DOUBLEBUF | OPENGL)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (1024 / 800), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glTranslatef(0.0, 0.0, -5)
    font = pygame.font.SysFont('Sans', 32)
    title_font = pygame.font.SysFont('Sans', 48, bold=True)
    pygame.display.set_caption('Лабораторная работа №6. Кузнецова Дарина')
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.4, 0.7, 0.2, 1))
    glLightfv(GL_LIGHT0, GL_POSITION, (0.0, 0.0, 1.0, 1.0))
    glLightf(GL_LIGHT0, GL_SPOT_CUTOFF, 30)
    glLightfv(GL_LIGHT0, GL_SPECULAR, (0, 0, 0, 1))
    glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, (0.0, 0.0, -1.0))
    glLightf(GL_LIGHT0, GL_SPOT_EXPONENT, 2.0)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    vertex = create_shader(GL_VERTEX_SHADER, """
    varying vec4 vertex_color;
                void main(){
                    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                    vertex_color = gl_Color;
                }""")

    fragment = create_shader(GL_FRAGMENT_SHADER, """
    varying vec4 vertex_color;
                void main() {
                    gl_FragColor = vertex_color;
    }""")

    # program = glCreateProgram()
    # glAttachShader(program, vertex)
    # glAttachShader(program, fragment)
    # glLinkProgram(program)
    # glUseProgram(program)


    glScalef(scale, scale, scale)
    run = True
    x_rotation = False
    y_rotation = False
    z_rotation = False
    while run:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                run = False
            elif ev.type == pygame.KEYDOWN:
                sign = 1
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    sign = -1

                if ev.key == pygame.K_UP:
                    vectors[0] = (vectors[0][0], vectors[0][1] + 10)
                elif ev.key == pygame.K_s:
                    run = False
                elif ev.key == pygame.K_x:
                    x_rotation = True
                elif ev.key == pygame.K_z:
                    z_rotation = True
                elif ev.key == pygame.K_y:
                    y_rotation = True
                elif ev.key == pygame.K_a:
                    if (note > 3 or sign > 0):
                        note = note + 1 * (sign)
                        models[0] = (*hyperbola_3d(note, 1), "Hyperbola %s" % note)
                        next_model(0)
                    else:
                        pass
                elif ev.key == pygame.K_i:
                    show_indices = not show_indices
                elif ev.key == pygame.K_n:
                    show_normals = not show_normals
                elif ev.key == pygame.K_m:
                    next_model(sign)
            elif ev.type == pygame.KEYUP:
                if ev.key == pygame.K_x:
                    x_rotation = False
                elif ev.key == pygame.K_z:
                    z_rotation = False
                elif ev.key == pygame.K_y:
                    y_rotation = False
                elif ev.key == pygame.K_RIGHT:
                    scale = 2.
                    glScalef(scale, scale, scale)
                elif ev.key == pygame.K_LEFT:
                    scale = 0.5
                    glScalef(scale, scale, scale)

        if not run:
            break
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        if x_rotation:
            glRotatef(sign * angle * 360, 1, 0, 0)
        if y_rotation:
            glRotatef(sign * angle * 360, 0, 1, 0)
        if z_rotation:
            glRotatef(sign * angle * 360, 0, 0, 1)

        draw_wire_GL()
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)
        pygame.display.flip()
    pygame.quit()

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(error)
        raise
    pygame.quit()
