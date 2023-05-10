from infer import Corrector

if __name__ == '__main__':
    inputs = [
        '老是较书。',
        '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
        '遇到一位很棒的奴生跟我聊天。',
        '遇到一位很美的女生跟我疗天。',
        '他们只能有两个选择：接受降新或自动离职。',
        '王天华开心得一直说话。'
    ]
    outputs = Corrector(inputs)
    for a, b in zip(inputs, outputs):
        print('input  :', a)
        print('predict:', b[0], b[1])
        print()

# output:
# input  : 老是较书。
# predict: 老师教书。 [('是', '师', 1, 2), ('较', '教', 2, 3)]

# input  : 感谢等五分以后，碰到一位很棒的奴生跟我可聊。
# predict: 感谢等五分以后，碰到一位很棒的女生跟我可聊。 [('奴', '女', 15, 16)]

# input  : 遇到一位很棒的奴生跟我聊天。
# predict: 遇到一位很棒的女生跟我聊天。 [('奴', '女', 7, 8)]

# input  : 遇到一位很美的女生跟我疗天。
# predict: 遇到一位很美的女生跟我聊天。 [('疗', '聊', 11, 12)]

# input  : 他们只能有两个选择：接受降新或自动离职。
# predict: 他们只能有两个选择：接受降薪或自动离职。 [('新', '薪', 13, 14)]

# input  : 王天华开心得一直说话。
# predict: 王天华开心得一直说话。 []