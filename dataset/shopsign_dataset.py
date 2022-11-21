import torch


class ShopsignDataset(torch.utils.data.Dataset):
    def __init__(self, gt_file):
        """
        Args:
            csv_file (string): Path to the csv file.
        """
        self.text, self.label = self.get_label(gt_file)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        x = self.text[idx]
        y = self.label[idx]

        return x, torch.tensor(int(y), dtype=torch.long)

    def get_label(self, gt_file):
        text = []
        label = []
        n = 0
        with open(gt_file, 'r', encoding='utf-8') as gt:
            lines = gt.readlines()
            for line in lines:
                print(line, n)
                n += 1
                text_i, label_i = line.split('\t')
                text.append(text_i)
                label.append(label_i)

        return text, label