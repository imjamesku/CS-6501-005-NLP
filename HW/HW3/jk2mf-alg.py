def parse(table):
    stack = []
    buffer = list(table)
    actions = []
    while buffer or len(stack) >= 2:
        #         print('buffer: {}'.format(buffer))
        #         print('stack: {}'.format(stack))
        #         print('actions: {}'.format(actions))
        #         print()
        if len(stack) >= 2:
            item2 = stack.pop()
            item1 = stack.pop()
            # item 1 is head
            if item1[0] == item2[2] and all(item[2] != item2[0] for item in buffer+stack):
                #             if item1[0] == item2[2]:
                stack.append(item1)
                actions.append('RIGHTARC')
            elif item1[2] != '0' and item2[0] == item1[2] and all(item[2] != item1[0] for item in buffer+stack):
                #             elif item1[2] != '0' and item2[0] == item1[2]:
                stack.append(item2)
                actions.append('LEFTARC')
            else:
                if buffer:
                    stack.extend([item1, item2, buffer.pop(0)])
                    actions.append('SHIFT')
                else:
                    return ['non-projective tree']
        elif buffer:
            stack.append(buffer.pop(0))
            actions.append('SHIFT')
        else:
            return ['non-projective tree']
    return actions


if __name__ == "__main__":
    tree = [['1', 'Book', '0'],
            ['2', 'me', '1'],
            ['3', 'the', '5'],
            ['4', 'morning', '5'],
            ['5', 'flight', '1']]
    actions = parse(tree)
    print(actions)
