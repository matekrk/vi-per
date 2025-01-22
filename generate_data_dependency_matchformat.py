import json
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# for MatplotlibDeprecationWarning: The tostring_rgb function was deprecated in Matplotlib 3.8 and will be removed two minor releases later. Use buffer_rgba instead.

def default_dependency(M):
    if len(M) == 1:
        return [[1.0]]
    if len(M) == 4:
        p_o = [0.5]
        p_sq = [0.75, 0.25]
        p_t = [0.67, 0.67, 0.33, 0.33]
        p_st = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        return [p_o, p_sq, p_t, p_st]

def get_current_probability(so_far, probs):
    if so_far == []:
        return probs[0][0]
    elif len(so_far) == 1:
        return probs[1][so_far[0]]
    elif len(so_far) == 2:
        return probs[2][so_far[0] + 2*so_far[1]]
    elif len(so_far) == 3:
        return probs[3][so_far[0] + 2*so_far[1] + 4*so_far[2]]

def create_artificialshapes_dependencies_dataset(N, M, img_size, datadir, datatxt, labeltxt, targettxt, no_overlap=False, coloured_figues=False, coloured_background=False, dependencies=None):
    
    if isinstance(M, int):
        M = list(range(M))

    dataset = []
    labels = []
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'pink', 'brown'] if coloured_figues else ['black']
    shapes = ['disk', 'square', 'triangle', 'star', 'hexagon', 'pentagon'] # ['o', 's', '^', '*', 'H', 'p']
    shapes = [shapes[i] for i in M]
    probs_dependencies = default_dependency(M) if dependencies is None else dependencies
    assert len(probs_dependencies) == len(M)

    os.makedirs(datadir, exist_ok=True)
    data_file = open(datatxt, 'w')
    label_file = open(labeltxt, 'w')
    target_file = open(targettxt, 'w')
    for shape in shapes:
        label_file.write(str(2) + f";{shape};{shape}" + '\n') # TODO: Support also counter for later
        label_file.write(f"no_{shape}" + ';' + f"there_isno_{shape}" + '\n')
        label_file.write(f"yes_{shape}" + ';' + f"there_is_{shape}" + '\n')

    so_far_stats = {}
    for image_id in tqdm(range(N)):
        dpi = 100
        fig, ax = plt.subplots(figsize=(img_size*(1/dpi), img_size*(1/dpi)), dpi=dpi)
        plt.axis('off')
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)

        if coloured_background:
            ax.set_facecolor(np.random.choice(colors))
        else:
            ax.set_facecolor('white')

        ax.axis('equal')

        lst_label = []
        lst_shapes = []

        so_far = []
        for shape in shapes:
            rand = np.random.rand()
            if rand > get_current_probability(so_far, probs_dependencies):
                so_far.append(0)
                continue
            color = np.random.choice(colors)
            size = np.random.randint(int(0.25*img_size), int(0.6*img_size))
            x = np.random.randint(0, img_size - size)
            y = np.random.randint(0, img_size - size)

            if shape == 'disk':
                circle = patches.Circle((x + size/2, y + size/2), size/2, edgecolor=color, facecolor=color, alpha=0.5)
                lst_shapes.append(circle)
                lst_label.append((shape, color, size, x, y))
                so_far.append(1)
            elif shape == 'square':
                square = patches.Rectangle((x, y), size, size, edgecolor=color, facecolor=color, alpha=0.5)
                lst_shapes.append(square)
                lst_label.append((shape, color, size, x, y))
                so_far.append(1)
            elif shape == 'triangle':
                triangle = patches.RegularPolygon((x + size/2, y + size/2), 3, radius=size/2, orientation=np.random.rand()*np.pi, edgecolor=color, facecolor=color, alpha=0.5)
                lst_shapes.append(triangle)
                lst_label.append((shape, color, size, x, y))
                so_far.append(1)
            elif shape == 'star':
                angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
                x_coords = np.cos(angles) * size/3
                y_coords = np.sin(angles) * size/3
                x_coords[1::2] *= size/16
                y_coords[1::2] *= size/16
                x_coords += x + size/3
                y_coords += y + size/3
                star = patches.Polygon(np.column_stack([x_coords, y_coords]), edgecolor=color, facecolor=color, alpha=0.5)
                lst_shapes.append(star)
                lst_label.append((shape, color, size, x, y))
                so_far.append(1)
            elif shape == 'hexagon':
                hexagon = patches.RegularPolygon((x + size/2, y + size/2), 6, radius=size/2, orientation=np.random.rand()*np.pi, edgecolor=color, facecolor=color, alpha=0.5)
                lst_shapes.append(hexagon)
                lst_label.append((shape, color, size, x, y))
                so_far.append(1)
            elif shape == 'pentagon':
                pentagon = patches.RegularPolygon((x + size/2, y + size/2), 5, radius=size/2, orientation=np.random.rand()*np.pi, edgecolor=color, facecolor=color, alpha=0.5)
                lst_shapes.append(pentagon)
                lst_label.append((shape, color, size, x, y))

        so_far_stats[tuple(so_far)] = so_far_stats.get(tuple(so_far), 0) + 1

        if no_overlap:
            new_shapes = []
            new_label = []
            for i, shape in enumerate(lst_shapes):
                overlaps = False
                for j in range(i+1, len(shapes)):
                    if check_overlap(shape, shapes[j]):
                        overlaps = True
                        break
                if not overlaps:
                    new_shapes.append(shape)
                    new_label.append(lst_label[i])
            lst_shapes = new_shapes
            lst_label = new_label

        for single_shape in lst_shapes:
            ax.add_patch(single_shape)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) # warning 
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        dataset.append(data)
        labels.append(lst_label)
        image_file = os.path.join(datadir, f'image_{image_id}.jpg')
        plt.savefig(image_file)
        plt.close(fig)

        shape_labels = [0] * len(shapes)
        for shape, _, _, _, _ in lst_label:
            shape_labels[shapes.index(shape)] = 1

        data = {
            'box': {'y': 0, 'x': 0, 'w': img_size, 'h': img_size},
            'box_id': f"e{image_id}x",
            'image_id': f"e{image_id}x",
            'image_file': image_file,
            'id': [f"yes_{shape}" if label == 1 else f"no_{shape}" for shape, label in zip(shapes, shape_labels)],
            'size': {'width': img_size, 'height': img_size}
        }
        data_file.write(json.dumps(data) + '\n')
        target_file.write(json.dumps(lst_label) + '\n')
    
    print(so_far_stats)

    data_file.close()
    label_file.close()
    target_file.close()
    return np.array(dataset), labels

def load_artificial_shapes_dataset(datadir, datatxt, labeltxt, targettxt):
    dataset = []
    labels = []
    with open(datatxt, 'r') as f:
        for line in f:
            data = json.loads(line)
            image_file = os.path.join(datadir, data['image_file'])
            dataset.append(plt.imread(image_file))
    # with open(labeltxt, 'r') as f:
    #     for line in f:
    #         labels.append(line.strip().split(';'))
    with open(targettxt, 'r') as f:
        for line in f:
            labels.append(json.loads(line))
    return np.array(dataset), labels


def check_overlap(shape1, shape2):
    poly1 = shape1.get_path().to_polygons()[0]
    poly2 = shape2.get_path().to_polygons()[0]
    return Polygon(poly1).intersects(Polygon(poly2))


def main():

    size =  64
    N = 10000 # 10000 7500
    M = 4 # 4 [4]
    main_dir = f"/shared/sets/datasets/vision/artificial_shapes/dependencies"

    coloured_background = False
    coloured_figues = True
    no_overlap = False
    dependencies = None

    def get_appendix(coloured_background, coloured_figues, no_overlap):
        def get_bool_str(v):
            return "T" if v else "F"
        return f"cb{get_bool_str(coloured_background)}_cf{get_bool_str(coloured_figues)}_no{get_bool_str(no_overlap)}"

    path_to_save = os.path.join(main_dir, f"dependenciesDEF_" + f"size{size}_" + f"len{N}_" + get_appendix(coloured_background, coloured_figues, no_overlap))

    datasetdir = os.path.join(path_to_save, "images")
    datasettxt = os.path.join(path_to_save, "data.txt")
    labelstxt = os.path.join(path_to_save, "label.txt")
    targettxt = os.path.join(path_to_save, "target.txt")
    print(f"Your data path will be: {path_to_save}")

    dataset, labels = create_artificialshapes_dependencies_dataset(N, M, size, datasetdir, datasettxt, labelstxt, targettxt, no_overlap, coloured_figues, coloured_background, dependencies)
    
    # for inspection
    # datasetnpy = os.path.join(path_to_save, "data.npy")
    # np.save(datasetnpy, dataset)
    # labelsnpy = os.path.join(path_to_save, "label.npy")
    # np.save(labelsnpy, labels)

if __name__ == "__main__":
    main()
