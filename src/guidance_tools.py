import guidance
from guidance import gen
from guidance_web_search import load_guidance_llama_cpp


@guidance
def add(lm, input1, input2):
    lm += f' = {int(input1) + int(input2)}'
    return lm

@guidance
def subtract(lm, input1, input2):
    lm += f' = {int(input1) - int(input2)}'
    return lm

@guidance
def multiply(lm, input1, input2):
    lm += f' = {float(input1) * float(input2)}'
    return lm

@guidance
def divide(lm, input1, input2):
    lm += f' = {float(input1) / float(input2)}'
    return lm


import weight_table

weight_table.load_weight_table()

lm = load_guidance_llama_cpp('small')
o = lm + '''\
1 + 1 = add(1, 1) = 2
2 - 3 = subtract(2, 3) = -1
'''
output = o + gen(max_tokens=15, tools=[add, subtract, multiply, divide])

print(output)

