import os
import time
import logging
import collections
import numpy as np
from PIL import Image
import dominate
import torch

class WebVisualizer():
    def __init__(self, opt):
        self.opt = opt
        self.display_id = opt.display_id
        self.win_size = opt.display_winsize
        self.use_html = (opt.html and (opt.mode == "Train"))
        self.name = opt.name
        self.saved = False
        self.type2id = {"Loss": 0, "Accuracy": 1, "Precision": 2, "Recall": 3, "F1_score": 4, "MeanLoss": 5, "MeanAccuracy": 6, "GradientBase": 7, "GradientAll": 8, "Other": 9}
        self.phase2id = {"Train": 0, "Validate": 1, "Test": 2}
        
        def ManualType():
            return collections.defaultdict(list)
        # store all the points for regular backup 
        self.plot_data = collections.defaultdict(ManualType)
        # line window info 
        self.win_info = collections.defaultdict(ManualType)
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)
        
        if self.use_html:
            self.web_dir = os.path.join(opt.model_dir, "web")
            self.img_dir = os.path.join(opt.model_dir, "image")
            print("Create web directory %s ..." %(self.web_dir))
            mkdirs([self.web_dir, self.img_dir])
            

    def reset(self):
        self.saved = False
    
    """
    type:  [Accuracy | Loss | Precision | Recall | F1_score | MeanLoss | MeanAccuracy | GradientBase | GradientAll | Other]
    phase: [Train | Validate | Test]
    """
    def plot_points(self, x, y, data_type, phase):
        line_name = data_type + "@" + phase
        self.plot_data[data_type][phase].append((x,y))
        # draw ininial line objects if not initialized
        if len(self.win_info[data_type][phase]) == 0:
            len_plotted = len(y)

            for index in range(len_plotted):
                win_id = self.type2id[data_type]*self.opt.class_num + index
                win = self.vis.line(X=np.array([x]),
                                    Y=np.array([y[index]]),
                                    opts=dict(
                                        title=data_type + (f" of Attribute {index}" if len_plotted == self.opt.class_num else "") + " Over Time",
                                        xlabel="epoch",
                                        ylabel=data_type,
                                        showlegend=True,
                                        width=900,
                                        height=450),
                                    win=win_id,
                                    name=line_name)
                self.win_info[data_type][phase].append(win)
        
        for index, value in enumerate(y): 
            win_id = self.win_info[data_type][phase][index] 
            self.vis.line(X=np.array([x]),
                          Y=np.array([value]),
                          win=win_id,
                          name=line_name,
                          update="append")
    
    def plot_images(self, image_dict, start_display_id, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.opt.image_ncols
            if ncols > 0:
                h, w = next(iter(image_dict.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(image_dict.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in image_dict.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=start_display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=start_display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in image_dict.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=start_display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image_numpy in image_dict.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                save_image(image_numpy, img_path)
            # update website
            webpage = HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in image_dict.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def backup(self, name):
        pass

    def test(self):
        pass

class HTML:
    def __init__(self, web_dir, title, reflesh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        # print(self.img_dir)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                dominate.tags.meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            dominate.tags.h3(str)

    def add_table(self, border=1):
        self.t = dominate.tags.table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=400):
        self.add_table()
        with self.t:
            with dominate.tags.tr():
                for im, txt, link in zip(ims, txts, links):
                    with dominate.tags.td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with dominate.tags.p():
                            with dominate.tags.a(href=os.path.join('images', link)):
                                dominate.tags.img(style="width:%dpx" % width, src=os.path.join('images', im))
                            dominate.tags.br()
                            dominate.tags.p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()

def print_loss(loss_list, label, epoch=0, batch_iter=0):
    if label == "Test":
        logging.info("[ %s Loss ] of Test Dataset:" % (label))
    else:
        logging.info("[ %s Loss ] of Epoch %d Batch %d" % (label, epoch, batch_iter))
    
    for index, loss in enumerate(loss_list):
        logging.info("----Attribute %d:  %f" %(index, loss))

def print_accuracy(accuracy_list, label, epoch=0, batch_iter=0):
    if label == "Test":
        logging.info("[ %s Accu ] of Test Dataset:" % (label))
    else:
        logging.info("[ %s Accu ] of Epoch %d Batch %d" %(label, epoch, batch_iter))
    
    for index, item in enumerate(accuracy_list):
        for top_k, value in item.items():
            logging.info("----Attribute %d Top%d: %f" %(index, top_k, value["ratio"]))

def print_metrics(metrics_list, label, epoch=0, batch_iter=0):
    if label == "Test":
        logging.info("[ %s Metrics ] of Test Dataset:" % (label))
    else:
        logging.info("[ %s Metrics ] of Epoch %d Batch %d" % (label, epoch, batch_iter))
    
    for index, item in enumerate(metrics_list):
        for top_k, value in item.items():
            logging.info("----Attribute %d Top%d:" % (index, top_k))
            logging.info("    Accuracy: %f" % value["accuracy"])
            logging.info("    Precision: %f" % value["precision"])
            logging.info("    Recall: %f" % value["recall"])
            logging.info("    F1 Score: %f" % value["f1_score"])

def tensor2im(image_tensor, mean, std, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = image_numpy.transpose(1, 2, 0)
    image_numpy *= std
    image_numpy += mean
    image_numpy *= 255.0
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def opt2file(opt, dst_file):
    args = vars(opt) 
    with open(dst_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print("%s: %s" %(str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        print('-------------- End ----------------')

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rmdir(path):
    if os.path.exists(path):
        os.system('rm -rf ' + path)

def load_label(label_file):
    rid2name = list()   # rid: real id, same as the id in label.txt
    id2rid = list()     # id: number from 0 to len(rids)-1 corresponding to the order of rids
    rid2id = list()     
    with open(label_file) as l:
        rid2name_dict = collections.defaultdict(str)
        id2rid_dict = collections.defaultdict(str)
        rid2id_dict = collections.defaultdict(str)
        new_id = 0 
        for line in l.readlines():
            line = line.strip('\n\r').split(';')
            if len(line) == 3: # attr description
                if len(rid2name_dict) != 0:
                    rid2name.append(rid2name_dict)
                    id2rid.append(id2rid_dict)
                    rid2id.append(rid2id_dict)
                    rid2name_dict = collections.defaultdict(str)
                    id2rid_dict = collections.defaultdict(str)
                    rid2id_dict = collections.defaultdict(str)
                    new_id = 0
                rid2name_dict["__name__"] = line[2]
                rid2name_dict["__attr_id__"] = line[1]
            elif len(line) == 2: # attr value description
                rid2name_dict[line[0]] = line[1]
                id2rid_dict[new_id] = line[0]
                rid2id_dict[line[0]] = new_id
                new_id += 1
        if len(rid2name_dict) != 0:
            rid2name.append(rid2name_dict)
            id2rid.append(id2rid_dict)
            rid2id.append(rid2id_dict)
    return rid2name, id2rid, rid2id

class MultiBinaryBCELoss(torch.nn.Module):
    def __init__(self, weight=None, reduction="mean"):
        super(MultiBinaryBCELoss, self).__init__()
        self.weight = torch.tensor(weight) if weight is not None else None
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to inputs
        # inputs = torch.sigmoid(inputs)
        targets = torch.stack(targets).reshape_as(inputs).to(dtype=torch.int32)
        
        # Compute individual binary cross-entropy losses
        bce_loss = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        
        # Take mean/sum of the losses for each data point
        if self.reduction == "mean":
            loss = bce_loss.mean(dim=0)
        elif self.reduction == "sum":
            loss = bce_loss.sum(dim=0)
        else:
            assert self.reduction is None
            loss = bce_loss

        # Apply weights if provided
        if self.weight is not None:
            loss = loss * self.weight.to(loss.device)

        # loss is of shape n_attributes
        return loss
