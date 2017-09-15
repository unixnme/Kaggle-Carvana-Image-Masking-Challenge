import multiprocessing


def createpdf(data):
    return ("This is my pdf data: %s\n" % data, 0)

def test1():
    data = [ "My data", "includes", "strings and", "numbers like", 42, "and", 3.14]
    number_of_processes = 5
    results = multiprocessing.Pool(number_of_processes).map(createpdf, data)
    outputs = [result[0] for result in results]
    pdfoutput = "".join(outputs)
    print pdfoutput

def test2():
    import train
    gen = train.train_generator()
    x,y = next(gen)
    a,b = next(gen)
    pass

if __name__ == '__main__':
    test2()