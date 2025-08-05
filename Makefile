C_SOURCES = $(wildcard matrix/*.c neuron/*.c utilities/*.c *.c)
HEADERS = $(wildcard matrix/*.h neuron/*.h utilities/*.h *.h)
OBJ = ${C_SOURCES:.c=.o}
CFLAGS = 

MAIN = main
CC = /usr/bin/gcc
LINKER = /usr/bin/ld

run: ${MAIN}
	./${MAIN}

main: ${OBJ}
	${CC} ${CFLAGS} $^ -o $@ -lm

# Generic rules
%.o: %.c ${HEADERS}
	${CC} ${CFLAGS} -c $< -o $@ -lm

clean:
	rm matrix/*.o *.o neuron/*.o utilities/*.o ${MAIN}
