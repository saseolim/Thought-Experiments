#pragma warning(disable : 4996)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main()
{
	int* (*p)[5] = (int*(*)[5])malloc(sizeof(int*) * 5);
	if (p == NULL)
	{
		perror("Failed to allcoate memory");
		return 1;
	}

	int a = 0;
	int b = 1;
	int c = 2;
	int d = 3;
	int e = 4;
	(*p)[0] = &a;
	(*p)[1] = &b;
	(*p)[2] = &c;
	(*p)[3] = &d;
	(*p)[4] = &e;

	for (int i = 0; i < 5; i++)
	{
		char c[10];
		sprintf(c, "%d", *((*p)[i]));

		printf("%s\n", c);
	}

	printf("\n");

	int* (*(*p2)[3])[5] = (int* (*(*)[3])[5])malloc(sizeof(int* (*)[5]) * 3);
	if (p2 == NULL)
	{
		perror("Failed to allcoate memory");
		return 2;
	}

	(*p2)[0] = p;
	(*p2)[1] = p;
	(*p2)[2] = p;

	for (int i = 0; i < 3; i++)
	{
		for (int j  = 0; j < 5; j++)
		{
			char c[10];
			sprintf(c, "%d", *(*(*p2)[i])[j]);
			printf("%s", c);
		}
		printf("\n");
	}

	printf("\n");

	free(p2);
	free(p);

	printf("end");
	fflush(stdout);

	return 0;
}