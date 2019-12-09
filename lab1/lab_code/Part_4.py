from lab_code import Part_1


class Node:

    def __init__(self, is_word=False, char='', init_list_size=60):
        self.char = char
        self.is_word = is_word
        self.now_words = 0  # 表示填充的字数
        self.child_list = [None] * init_list_size

    def add_child(self, child):
        if self.now_words / float(len(self.child_list)) > float(2 / 3):
            self.now_words = 0
            self.rehash(child)
        index = self.hash_char(char=child.char)
        while self.child_list[index] is not None:
            index = (index + 1) % len(self.child_list)
        self.child_list[index] = child
        self.now_words += 1

    def get_node_by_char(self, char):
        index = self.hash_char(char)
        while True:
            child = self.child_list[index]
            if child is None:
                return None
            if child.char == char:
                return child
            index = (index + 1) % len(self.child_list)

    def hash_char(self, char):
        return ord(char) % len(self.child_list)

    def rehash(self, child):
        old_child_list = self.child_list
        self.child_list = [None] * (2 * len(self.child_list))
        for every_child in old_child_list:
            if every_child is not None:
                index = self.hash_char(char=every_child.char)
                while self.child_list[index] is not None:
                    index = (index + 1) % len(self.child_list)
                self.child_list[index] = every_child
                self.now_words += 1
        self.add_child(child)


class DicAction:
    Words_List = []

    @staticmethod
    def get_fmm_dic(dic_path='../io_file/dic/dic.txt', choice=True):
        if choice:  # 为真表示需要再初始化词列表
            for line in open(dic_path, 'r', encoding='UTF-8'):
                DicAction.Words_List.append(line.split()[0])
        root = Node(init_list_size=7000)
        for word in DicAction.Words_List:
            DicAction.insert_fmm(word, root)
        return root

    @staticmethod
    def get_bmm_dic(dic_path='../io_file/dic/dic.txt', choice=True):
        if choice:  # 为真表示需要再初始化词列表
            for line in open(dic_path, 'r', encoding='UTF-8'):
                DicAction.Words_List.append(line.split()[0])
        root = Node(init_list_size=7000)
        for word in DicAction.Words_List:
            DicAction.insert_bmm(word, root)
        return root

    @staticmethod
    def insert_fmm(word, root):
        length = len(word)
        count = 1
        node = root.get_node_by_char(word[0])
        before_node = root
        while node is not None:
            if count == length:
                node.is_word = True
                return
            before_node = node
            node = node.get_node_by_char(word[count])
            count += 1
        count -= 1
        while count < length:
            node = Node()
            node.char = word[count]
            count += 1
            before_node.add_child(node)
            before_node = node
        node.is_word = True

    @staticmethod
    def insert_bmm(word, root):
        count = len(word) - 1
        node = root.get_node_by_char(word[count])
        before_node = root
        while node is not None:
            if count == 0:
                node.is_word = True
                return
            count -= 1
            before_node = node
            node = node.get_node_by_char(word[count])
        while count >= 0:
            node = Node()
            node.char = word[count]
            count -= 1
            before_node.add_child(node)
            before_node = node
        node.is_word = True


class StrMatch:
    @staticmethod
    def fmm(root, txt_path=Part_1.Test_File, fmm_path='../io_file/seg/seg_fmm_1.txt'):
        seg_result = ''
        file = open(txt_path, 'r', encoding='utf-8')
        for line in file:
            seg_line, line = '', line[:len(line) - 1]  # 去掉读取的换行符
            while len(line) > 0:
                count = 0
                terminal_word = line[0]
                node = root.get_node_by_char(line[0])
                while node is not None:
                    count += 1
                    if node.is_word:
                        terminal_word = line[:count]
                    if count == len(line):
                        break
                    node = node.get_node_by_char(line[count])
                line = line[len(terminal_word):]
                seg_line += terminal_word + '/ '
            seg_result += Part_1.pre_line(seg_line) + '\n'
        open(fmm_path, 'w', encoding='UTF-8').write(seg_result)

    @staticmethod
    def bmm(root, txt_path=Part_1.Test_File, bmm_path='../io_file/seg/seg_bmm_1.txt'):
        seg_result = ''
        file = open(txt_path, 'r', encoding='utf-8')
        for line in file:
            seg_list = []
            line = line[:len(line) - 1]
            while len(line) > 0:
                count = len(line) - 1
                terminal_word = line[count]
                node = root.get_node_by_char(line[count])
                while node is not None:
                    if node.is_word:
                        terminal_word = line[count:]
                    count -= 1
                    if count < 0:
                        break
                    node = node.get_node_by_char(line[count])
                line = line[:len(line) - len(terminal_word)]
                seg_list.insert(0, terminal_word + '/ ')
            seg_result += Part_1.pre_line(''.join(seg_list)) + '\n'
        open(bmm_path, 'w', encoding='UTF-8').write(seg_result)
