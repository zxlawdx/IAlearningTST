import temp as tmp
# import raiz as rz
def converter(entrada):
    tmp.entrada_array = tmp.np.array([entrada], dtype=float)
    tmp.resultado = tmp.modelo.predict(tmp.entrada_array)[0][0]   
    return tmp.resultado

# def raiz(entrada):
#     rz.entrada_array = rz.np.array([entrada], dtype=float)
#     rz.resultado = rz.modelo.predict(rz.entrada_array)[0][0]
#     return rz.resultado

# entrada = float(input("digite um valor: "))
# raiz(entrada)
# print(rz.resultado)