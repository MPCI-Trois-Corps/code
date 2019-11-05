np = int(input())
poids = input()
Poids = []
L = []

for i in range(len(poids)):
    if poids[i] == " ":
        L.append(i)
L.append(len(poids))

a,carburant = 0,0
for k in L:
    if a == k :
        None
    else :
        if int(poids[a:k]) <= 90 :
            carburant += 60
        else :
            carburant += 80
    a = k+1
print(carburant)