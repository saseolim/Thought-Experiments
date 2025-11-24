using Jacobian_Test;

namespace Tensor
{
    internal class Program
    {
        static void Tensor2_Test()
        {
            Tensor2<decimal> tensor1 = new(new int[] { 2, 3 });
            Tensor2<decimal> tensor2 = new(new int[] { 2, 3 });

            tensor2 += 1;

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    tensor1.Set(new int[] { i, j }, i * 2 + j + 1);
                }
            }

            Console.WriteLine(tensor1.ToString2D());
            Console.WriteLine(tensor2.ToString2D());

            Tensor2<decimal> tensor3 = Tensor2<decimal>.Jacobian((x) => x * x, tensor1, 0.00001m);

            Tensor2<decimal> tensor4 = Tensor2<decimal>.Chain(tensor1, 0.00001m, (x) => 2 * x, (x) => x * x);

            Console.WriteLine(tensor3.ToString());
            Console.WriteLine(tensor4.ToString());
        }
        static void Main(string[] args)
        {
            Tensor2_Test();
        }
    }
}
