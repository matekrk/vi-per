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

def create_artificialshapes_dataset(N, img_size, datadir, datatxt, labeltxt, no_overlap=False, coloured_figues=False, coloured_background=False, bias_classes=None, simplicity=0):
    dataset = []
    labels = []
    colors = ['red' , 'green', 'blue', 'yellow', 'cyan', 'magenta'] if coloured_figues else ['cyan']
    shapes = ['disk', 'square', 'triangle', 'star', 'hexagon', 'pentagon']
    bias_classes = [1/(len(shapes)) for _ in shapes] if bias_classes is None else bias_classes
    assert len(bias_classes) == len(shapes)

    def get_prob_figures(simplicity):
        match simplicity:
            case 0:
                return [0.1, 0.3, 0.25, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025]
            case 1:
                return [0.34, 0.33, 0.33]
            case 2:
                return [0.2, 0.5, 0.2, 0.1]
            case 3:
                return [0.5, 0.5]
            case 4:
                return [0.0, 1.0]
            case 5:
                return [1.0]
    probs_number_figures = get_prob_figures(simplicity)

    os.makedirs(datadir, exist_ok=True)
    data_file = open(datatxt, 'w')
    label_file = open(labeltxt, 'w')
    for shape in shapes:
        label_file.write(str(2) + f";{shape};{shape}" + '\n') # TODO: Support also counter for later
        label_file.write(f"no_{shape}" + ';' + f"there_isno_{shape}" + '\n')
        label_file.write(f"yes_{shape}" + ';' + f"there_is_{shape}" + '\n')


    for image_id in tqdm(range(N)):
        fig, ax = plt.subplots(figsize=(img_size*0.1, img_size*0.1), dpi=10)
        plt.axis('off')
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)

        if coloured_background:
            ax.set_facecolor(np.random.choice(colors))
        else:
            ax.set_facecolor('white')

        ax.axis('equal')

        num_shapes = np.random.choice(range(0,len(probs_number_figures)), p=probs_number_figures)
        lst_label = []
        lst_shapes = []

        for _ in range(num_shapes):
            shape = np.random.choice(shapes)
            color = np.random.choice(colors)
            size = np.random.randint(int(0.15*img_size), int(0.6*img_size))
            x = np.random.randint(0, img_size - size)
            y = np.random.randint(0, img_size - size)

            if shape == 'disk':
                circle = patches.Circle((x + size/2, y + size/2), size/2, edgecolor=color, facecolor=color, alpha=0.5)
                lst_shapes.append(circle)
                lst_label.append((shape, color, size, x, y))
            elif shape == 'square':
                square = patches.Rectangle((x, y), size, size, edgecolor=color, facecolor=color, alpha=0.5)
                lst_shapes.append(square)
                lst_label.append((shape, color, size, x, y))
            elif shape == 'triangle':
                triangle = patches.RegularPolygon((x + size/2, y + size/2), 3, radius=size/2, orientation=np.random.rand()*np.pi, edgecolor=color, facecolor=color, alpha=0.5)
                lst_shapes.append(triangle)
                lst_label.append((shape, color, size, x, y))
            elif shape == 'star':
                angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
                x_coords = np.cos(angles) * size/4
                y_coords = np.sin(angles) * size/4
                x_coords[1::2] *= size/4
                y_coords[1::2] *= size/4
                x_coords += x + size/2
                y_coords += y + size/2
                star = patches.Polygon(np.column_stack([x_coords, y_coords]), edgecolor=color, facecolor=color, alpha=0.5)
                lst_shapes.append(star)
                lst_label.append((shape, color, size, x, y))
            elif shape == 'hexagon':
                hexagon = patches.RegularPolygon((x + size/2, y + size/2), 6, radius=size/2, orientation=np.random.rand()*np.pi, edgecolor=color, facecolor=color, alpha=0.5)
                lst_shapes.append(hexagon)
                lst_label.append((shape, color, size, x, y))
            elif shape == 'pentagon':
                pentagon = patches.RegularPolygon((x + size/2, y + size/2), 5, radius=size/2, orientation=np.random.rand()*np.pi, edgecolor=color, facecolor=color, alpha=0.5)
                lst_shapes.append(pentagon)
                lst_label.append((shape, color, size, x, y))

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
    
    data_file.close()
    label_file.close()

    return np.array(dataset), labels

def check_overlap(shape1, shape2):
    poly1 = shape1.get_path().to_polygons()[0]
    poly2 = shape2.get_path().to_polygons()[0]
    return Polygon(poly1).intersects(Polygon(poly2))


def main():

    size =  64
    N = 10240
    main_dir = f"/shared/sets/datasets/vision/artificial_shapes/"

    coloured_background = False
    coloured_figues = True
    no_overlap = False
    bias_classes = None
    simplicity = 3

    def get_appendix(coloured_background, coloured_figues, no_overlap):
        def get_bool_str(v):
            return "T" if v else "F"
        return f"cb{get_bool_str(coloured_background)}_cf{get_bool_str(coloured_figues)}_no{get_bool_str(no_overlap)}"

    path_to_save = os.path.join(main_dir, f"size{size}_" + f"simplicity{simplicity}_" + f"len{N}_" + get_appendix(coloured_background, coloured_figues, no_overlap))

    datasetdir = os.path.join(path_to_save, "images")
    datasettxt = os.path.join(path_to_save, "data.txt")
    labelstxt = os.path.join(path_to_save, "label.txt")
    print(f"Your data path will be: {path_to_save}")

    dataset, labels = create_artificialshapes_dataset(N, size, datasetdir, datasettxt, labelstxt, no_overlap, coloured_figues, coloured_background, bias_classes, simplicity)
    
    # for inspection
    # datasetnpy = os.path.join(path_to_save, "data.npy")
    # np.save(datasetnpy, dataset)
    # labelsnpy = os.path.join(path_to_save, "label.npy")
    # np.save(labelsnpy, labels)

if __name__ == "__main__":
    main()
