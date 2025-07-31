using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Dynamic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.Marshalling;
using System.Text;
using System.Threading.Tasks;
using TensorLib;

/*
 int[]방식으로 tensor구현하기로 결정
역전파 구현 예정
 */

namespace Jacobian_Test
{
    public class Tensor<T>
    {
        public Array tensor { get; private set; }
        public int Rank { get => Shape.Length ; }
        public int[] Shape { get; private set; }
        public bool IsJacobian { get; private set; }
        public int[] JacobianFuncShape { get; private set; }
        public int JF_Rank { get => JacobianFuncShape.Length; }
        public int[] JacobianParamsShape { get; private set; }
        public int JP_Rank { get => JacobianParamsShape.Length; }

        public Tensor(int[] Shape)
        {
            this.Shape = (int[])Shape.Clone();
            this.tensor = Array.CreateInstance(typeof(T), (int[])Shape.Clone());
            //딕셔너리 Dictionary<int[],T>고러바람

            this.IsJacobian = false;
            this.JacobianFuncShape = new int[0];
            this.JacobianParamsShape = new int[0];
        }
        public Tensor(int[] Shape, int[] JacobianFuncShape, int[] JacobianParamsShape)
        {
            this.Shape = Shape;
            this.tensor = Array.CreateInstance(typeof(T), Shape);

            this.IsJacobian = true;
            this.JacobianFuncShape = (int[])JacobianFuncShape.Clone();
            this.JacobianParamsShape = (int[])JacobianParamsShape.Clone();
        }
        public Tensor(Tensor<T> OneBone)
        {
            this.tensor = (Array)OneBone.tensor.Clone();
            this.Shape = (int[])OneBone.Shape.Clone();

            this.IsJacobian = OneBone.IsJacobian;
            this.JacobianFuncShape = (int[])OneBone.JacobianFuncShape.Clone();
            this.JacobianParamsShape = (int[])OneBone.JacobianParamsShape.Clone();
        }
        public Tensor(Tensor<T> OneBone, int[] JacobianFuncShape, int[] JacobianParamsShape)
        {
            this.tensor = (Array)OneBone.tensor.Clone();
            this.Shape = (int[])OneBone.Shape.Clone();

            this.IsJacobian = true;
            this.JacobianFuncShape = (int[])JacobianFuncShape.Clone();
            this.JacobianParamsShape = (int[])JacobianParamsShape.Clone();
        }

        public T Get(int[] index)
        {
            return (T)tensor.GetValue(index);
        }
        public void Set(int[] index, T tensor_component)
        {
            tensor.SetValue(tensor_component, index);
        }

        private Array CloneTensorArray(Array source)
        {
            var shape = new int[source.Rank];
            for (int i = 0; i < source.Rank; i++)
                shape[i] = source.GetLength(i);

            var clone = Array.CreateInstance(typeof(T), shape);

            // 모든 요소 복사
            var indices = new int[shape.Length];
            void Recurse(int dim)
            {
                if (dim == shape.Length)
                {
                    clone.SetValue(source.GetValue(indices), indices);
                    return;
                }

                for (int i = 0; i < shape[dim]; i++)
                {
                    indices[dim] = i;
                    Recurse(dim + 1);
                }
            }

            Recurse(0);
            return clone;
        }

        public Tensor<T> Clone()
        {
            Tensor<T> clone = new Tensor<T>(this.Shape);
            clone.tensor = CloneTensorArray(this.tensor);
            clone.IsJacobian = this.IsJacobian;
            clone.JacobianFuncShape = (int[])this.JacobianFuncShape.Clone();
            clone.JacobianParamsShape = (int[])this.JacobianParamsShape.Clone();
            return clone;
        }


        public T ToScalar()
        {
            List<int> Index = new();
            for (int i = 0; i < this.Rank; i++)
            {
                if (this.Shape[i] != 1)
                    throw new Exception();
                Index.Add(0);
            }

            return (T)this.Get(Index.ToArray());
        }

        public Tensor<T> Transpose2D()
        {
            var shape = new int[] { this.Shape[1], this.Shape[0] };
            var result = new Tensor<T>(shape);

            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    result.Set(new int[] { i, j }, this.Get(new int[] { j, i }));
                }
            }

            return result;
        }

        public string ToString2D()
        {
            if (this.Rank != 2)
                return "";

            string Out = "";

            for (int x = 0; x < this.Shape[0]; x++)
            {
                for (int y = 0; y < this.Shape[1]; y++)
                {
                    Out += $" {this.Get(new int[] { x, y })} ";
                }
                Out += "\n";
            }
            Out += "\n";
            return Out;
        }

        public string ToString1D()
        {
            if (this.Rank != 1)
                return "";

            string Out = "";

            for (int y = 0; y < this.Shape[0]; y++)
            {
                Out += $" {this.Get(new int[] { y })} ";
            }
            Out += "\n";
            return Out;
        }

        public override string ToString()
        {
            string Out = "";

            void Recurse(List<int> NowIndex, int depth)
            {
                if (depth == this.Rank)
                {
                    string Index = " ";
                    foreach (var s in NowIndex)
                    {
                        Index += $"{s} ";
                    }

                    Out += $"[{Index}] : {this.Get(NowIndex.ToArray())}\n";
                }
                else
                {
                    for (int i = 0; i < this.Shape[depth]; i++)
                    {
                        NowIndex.Add(i);
                        Recurse(NowIndex, depth + 1);
                        NowIndex.RemoveAt(NowIndex.Count - 1);
                    }
                }
            }
            Recurse(new List<int>(), 0);
            return Out;
        }

        public static Tensor<T> TensorOneOperation(Tensor<T> tensor,
            Func<T,T> function)
        {
            Tensor<T> Out = new(tensor.Shape);

            void Recurse(List<int> NowIndex, int depth)
            {
                if (depth == Out.Rank)
                {
                    Out.Set(
                        NowIndex.ToArray(),
                        function(tensor.Get(NowIndex.ToArray()))
                        );
                }
                else
                {
                    for (int i = 0; i < Out.Shape[depth]; i++)
                    {
                        NowIndex.Add(i);
                        Recurse(NowIndex, depth + 1);
                        NowIndex.RemoveAt(NowIndex.Count - 1);
                    }
                }
            }

            Recurse(new List<int>(), 0);
            return Out;
        }
        /**
        public struct OneTensor<T>
        {
            public Tensor<T> Tensor1;
            public Tensor<T> OutTensor;
            public OneTensor(Tensor<T> tensor1, Tensor<T> Out)
            {
                this.Tensor1 = tensor1;
                this.OutTensor = Out;
            }
        }
        public struct TwoTensor<T>
        {
            public Tensor<T> Tensor1;
            public Tensor<T> Tensor2;
            public Tensor<T> OutTensor;
            public TwoTensor(Tensor<T> tensor1, Tensor<T> tensor2, Tensor<T> Out)
            {
                this.Tensor1 = tensor1;
                this.Tensor2 = tensor2;
                this.OutTensor = Out;
            }
        }
        **/
        public static Tensor<T> TensorTwoOperation(Tensor<T> tensor1 , Tensor<T> tensor2,
            Func<T,T,T> function)
        {
            if (tensor1.Rank != tensor2.Rank)
                throw new ArgumentException("두 Tensor의 Rank가 일치하지 않습니다.");
            for(int i = 0; i < tensor1.Rank; i++)
            {
                if (tensor1.Shape[i] != tensor2.Shape[i])
                    throw new ArgumentException("두 Tensor의 Shape가 일치하지 않습니다.");
            }
            Tensor<T> Out = new(tensor1.Shape);

            void Recurse(List<int> NowIndex, int depth)
            {
                if (depth == Out.Rank)
                {
                    Out.Set(
                        NowIndex.ToArray(),
                        function(tensor1.Get(NowIndex.ToArray()), tensor2.Get(NowIndex.ToArray()))
                        );
                }
                else
                {
                    for (int i = 0; i < Out.Shape[depth]; i++)
                    {
                        NowIndex.Add(i);
                        Recurse(NowIndex, depth + 1);
                        NowIndex.RemoveAt(NowIndex.Count - 1);
                    }
                }
            }

            Recurse(new List<int>(), 0);
            return Out;
        }

        public static Tensor<T> TensorTwoOperation(Tensor<T> tensor, T K,
            Func<T, T, T> function)
        {
            Tensor<T> Out = new(tensor.Shape);

            void Recurse(List<int> NowIndex, int depth)
            {
                if (depth == Out.Rank)
                {
                    Out.Set(
                        NowIndex.ToArray(),
                        function(tensor.Get(NowIndex.ToArray()), K)
                        );
                }
                else
                {
                    for (int i = 0; i < Out.Shape[depth]; i++)
                    {
                        NowIndex.Add(i);
                        Recurse(NowIndex, depth + 1);
                        NowIndex.RemoveAt(NowIndex.Count - 1);
                    }
                }
            }

            Recurse(new List<int>(), 0);
            return Out;
        }

        public static Tensor<T> operator +(Tensor<T> A, Tensor<T> B)
        {
            return TensorTwoOperation(A, B,
                (a, b) =>
                {
                    return (dynamic)a + (dynamic)b;
                });
        }
        public static Tensor<T> operator +(Tensor<T> A, T K)
        {
            return TensorTwoOperation(A, K,
                (a, k) =>
                {
                    return (dynamic)a + (dynamic)k;
                });
        }
        public static Tensor<T> operator +(T K, Tensor<T> A)
        {
            return TensorTwoOperation(A, K,
                (a, k) =>
                {
                    return (dynamic)a + (dynamic)k;
                });
        }

        public static Tensor<T> operator -(Tensor<T> A, Tensor<T> B)
        {
            return TensorTwoOperation(A, B,
                (a,b) =>
                {
                     return (dynamic)a - (dynamic)b;
                });
        }
        public static Tensor<T> operator -(Tensor<T> A, T K)
        {
            return TensorTwoOperation(A, K,
                (a, k) =>
                {
                    return (dynamic)a - (dynamic)k;
                });
        }
        public static Tensor<T> operator -(T K, Tensor<T> A)
        {
            return TensorTwoOperation(A, K,
                (a, k) =>
                {
                    return (dynamic)a - (dynamic)k;
                });
        }

        public static Tensor<T> operator *(Tensor<T> A, Tensor<T> B)
        {
            return TensorTwoOperation(A, B,
                (a, b) =>
                {
                    return (dynamic)a * (dynamic)b;
                });
        }

        public static Tensor<T> operator *(Tensor<T> A, T K)
        {
            return TensorTwoOperation(A, K,
                (a, k) =>
                {
                    return (dynamic)a * (dynamic)k;
                });
        }

        public static Tensor<T> operator *(T K, Tensor<T> A)
        {
            return TensorTwoOperation(A, K,
                (a, k) =>
                {
                    return (dynamic)a * (dynamic)k;
                });
        }

        public static Tensor<T> operator /(Tensor<T> A, T K)
        {
            return TensorTwoOperation(A, K,
                (a,k) =>
                {
                    return (dynamic)a / (dynamic)k;
                });
        }

        public static decimal Sqrt(decimal x, decimal epsilon = 1e-13m)
        {
            if (x < 0) throw new ArgumentException("Cannot compute square root of a negative number");

            if (x == 0 || x == 1) return x;

            decimal guess = x / 2;
            while (Math.Abs(guess * guess - x) > epsilon)
            {
                guess = (guess + x / guess) / 2;
            }
            return guess;
        }

        public static decimal FrobeniusNorm(Tensor<T> tensor)
        {
            decimal sum = 0m;

            void Recurse(List<int> nowIndex, int depth)
            {
                if (depth == tensor.Rank)
                {
                    decimal value = (dynamic)tensor.Get(nowIndex.ToArray());
                    sum += value * value;
                }
                else
                {
                    for (int i = 0; i < tensor.Shape[depth]; i++)
                    {
                        nowIndex.Add(i);
                        Recurse(nowIndex, depth + 1);
                        nowIndex.RemoveAt(nowIndex.Count - 1);
                    }
                }
            }

            Recurse(new List<int>(), 0);
            return Sqrt(sum);
        }

        //함수들은 가장 나중에 연산되는 중첩 함수 순서대로
        //예시로 dz/dy * dy/df * df/dx일시 x, Epsilon, z(y), y(f), f(x) 순서대로 넣는다
        public static Tensor<T> Chain(Tensor<T> InputParamter, T Epsilon, params Func<Tensor<T>,Tensor<T>>[] Functions) 
        {
            List<List<int>> AllForShapes = new();
            AllForShapes.Add(new List<int>(InputParamter.Shape));
            List<Tensor<T>> Tensors = new();
            Tensors.Add(InputParamter);
            for (int i = Functions.Length - 1; i > -1; i--)
            {
                //Tensor<T> ThisShapeInput = new Tensor<T>(AllForShapes[AllForShapes.Count - 1].ToArray());
                //ThisShapeInput = Functions[i](ThisShapeInput);
                Tensor<T> ThisShapeInput = Tensors[Tensors.Count - 1];
                ThisShapeInput = Functions[i](ThisShapeInput);
                Tensors.Add(ThisShapeInput);
                AllForShapes.Add(new List<int>(ThisShapeInput.Shape));
            }
            AllForShapes.Reverse();
            Tensors.Reverse();

            List<Tensor<T>> Jacobians = new();
            Tensor<T> JacobianMemory = Tensors[Tensors.Count - 1];
            for (int i = Functions.Length - 1; i > -1; i--)
            {
                Jacobians.Add(Jacobian(Functions[i], JacobianMemory, Epsilon));
                JacobianMemory = Tensors[i];
                //JacobianMemory = Jacobians[Jacobians.Count - 1].Clone();
                //JacobianMemory = new Tensor<T>(AllForShapes[i].ToArray());
            }
            Jacobians.Reverse();

            Tensor<T> TensorDotMemonry = Jacobians[0];

            
            for (int i = 1; i < Jacobians.Count; i++)
            {
                //List<int> axisA = new();
                //List<int> axisB = new();

                //for (int ii = TensorDotMemonry.JF_Rank; ii < TensorDotMemonry.Rank; ii++)
                //    axisA.Add(ii);
                //for (int ii = 0; ii < Jacobians[i].JF_Rank; ii++)
                //    axisB.Add(ii);

                //정확한 축 매칭 (출력축 = 앞 텐서 func, 입력축 = 뒷 텐서 param)
                int JFM_A = TensorDotMemonry.JF_Rank;
                int JPP_B = Jacobians[i].JP_Rank;

                //var axisA = Enumerable.Range(TensorDotMemonry.JF_Rank, TensorDotMemonry.JP_Rank).ToList();
                //var axisB = Enumerable.Range(0, Jacobians[i].JF_Rank).ToList();

                int[] axisA = Enumerable.Range(TensorDotMemonry.JF_Rank, TensorDotMemonry.JP_Rank).ToArray(); // 앞: 입력 축
                int[] axisB = Enumerable.Range(0, Jacobians[i].JF_Rank).ToArray(); // 뒤: 출력 축


                TensorDotMemonry = new Tensor<T>(TensorDot(TensorDotMemonry, Jacobians[i], axisA.ToArray(), axisB.ToArray()), TensorDotMemonry.JacobianFuncShape, Jacobians[i].JacobianParamsShape);
            }

            return TensorDotMemonry;
        }

        public static Tensor<T> Jacobian(Func<Tensor<T>, Dictionary<string,Tensor<T>>, Tensor<T>> Function, Tensor<T> Parameter, Dictionary<string,Tensor<T>> StaticParmeter,
    T Epsilon)
        {
            List<int> OutShape = new();

            Tensor<T> F = Function(Parameter, StaticParmeter);

            for (int i = 0; i < F.Rank; i++)
                OutShape.Add(F.Shape[i]);
            for (int i = 0; i < Parameter.Rank; i++)
                OutShape.Add(Parameter.Shape[i]);

            Tensor<T> OneParameterMoveGetFunctionTensor = F.Clone();
            Tensor<T> result = new(OutShape.ToArray(), (int[])F.Shape.Clone(), (int[])Parameter.Shape.Clone());

            Tensor<T> PartialDifferential(Func<Tensor<T>, Dictionary<string,Tensor<T>>, Tensor<T>> Function, Tensor<T> Params, Tensor<T> Params_OneMove)
            {
                return (Function(Params_OneMove, StaticParmeter) - Function(Params, StaticParmeter)) / Epsilon;
            }

            void Recurse_JacobianSetF(List<int> NowIndex, int depth, int[] Index)
            {
                if (depth == F.Rank)
                {
                    List<int> JacobianIndex = new(NowIndex);
                    JacobianIndex.AddRange(Index);
                    result.Set(JacobianIndex.ToArray(),
                        OneParameterMoveGetFunctionTensor.Get(NowIndex.ToArray()));
                }
                else
                {
                    for (int i = 0; i < F.Shape[depth]; i++)
                    {
                        NowIndex.Add(i);
                        Recurse_JacobianSetF(NowIndex, depth + 1, Index);
                        NowIndex.RemoveAt(NowIndex.Count - 1);
                    }
                }
            }

            void Recurse_ParamsAccess(List<int> NowIndex, int depth)
            {
                if (depth == Parameter.Rank)
                {
                    Tensor<T> Parameter_OneMove = Parameter.Clone();
                    Parameter_OneMove.Set(
                        NowIndex.ToArray(),
                        (dynamic)Parameter_OneMove.Get(NowIndex.ToArray()) + Epsilon
                        );

                    OneParameterMoveGetFunctionTensor = PartialDifferential(Function, Parameter, Parameter_OneMove);

                    Recurse_JacobianSetF(new List<int>(), 0, NowIndex.ToArray());
                }
                else
                {
                    for (int i = 0; i < Parameter.Shape[depth]; i++)
                    {
                        NowIndex.Add(i);
                        Recurse_ParamsAccess(NowIndex, depth + 1);
                        NowIndex.RemoveAt(NowIndex.Count - 1);
                    }
                }
            }

            Recurse_ParamsAccess(new List<int>(), 0);
            return result;
        }

        public static Tensor<T> Jacobian(Func<Tensor<T>, List<Tensor<T>>, Tensor<T>> Function, Tensor<T> Parameter, List<Tensor<T>> StaticParmeter,
            T Epsilon)
        {
            List<int> OutShape = new();

            Tensor<T> F = Function(Parameter, StaticParmeter);

            for (int i = 0; i < F.Rank; i++)
                OutShape.Add(F.Shape[i]);
            for (int i = 0; i < Parameter.Rank; i++)
                OutShape.Add(Parameter.Shape[i]);

            Tensor<T> OneParameterMoveGetFunctionTensor = F.Clone();
            Tensor<T> result = new(OutShape.ToArray(), (int[])F.Shape.Clone(), (int[])Parameter.Shape.Clone());

            Tensor<T> PartialDifferential(Func<Tensor<T>, List<Tensor<T>>, Tensor<T>> Function, Tensor<T> Params, Tensor<T> Params_OneMove)
            {
                return (Function(Params_OneMove,StaticParmeter) - Function(Params,StaticParmeter)) / Epsilon;
            }

            void Recurse_JacobianSetF(List<int> NowIndex, int depth, int[] Index)
            {
                if (depth == F.Rank)
                {
                    List<int> JacobianIndex = new(NowIndex);
                    JacobianIndex.AddRange(Index);
                    result.Set(JacobianIndex.ToArray(),
                        OneParameterMoveGetFunctionTensor.Get(NowIndex.ToArray()));
                }
                else
                {
                    for (int i = 0; i < F.Shape[depth]; i++)
                    {
                        NowIndex.Add(i);
                        Recurse_JacobianSetF(NowIndex, depth + 1, Index);
                        NowIndex.RemoveAt(NowIndex.Count - 1);
                    }
                }
            }

            void Recurse_ParamsAccess(List<int> NowIndex, int depth)
            {
                if (depth == Parameter.Rank)
                {
                    Tensor<T> Parameter_OneMove = Parameter.Clone();
                    Parameter_OneMove.Set(
                        NowIndex.ToArray(),
                        (dynamic)Parameter_OneMove.Get(NowIndex.ToArray()) + Epsilon
                        );

                    OneParameterMoveGetFunctionTensor = PartialDifferential(Function, Parameter, Parameter_OneMove);

                    Recurse_JacobianSetF(new List<int>(), 0, NowIndex.ToArray());
                }
                else
                {
                    for (int i = 0; i < Parameter.Shape[depth]; i++)
                    {
                        NowIndex.Add(i);
                        Recurse_ParamsAccess(NowIndex, depth + 1);
                        NowIndex.RemoveAt(NowIndex.Count - 1);
                    }
                }
            }

            Recurse_ParamsAccess(new List<int>(), 0);
            return result;
        }
        /*
        public static Tensor<T> Jacobian(Func<Tensor<T>, Tensor<T>> Function, Tensor<T> Parameter, T Epsilon)
        {
            Tensor<T> F = Function(Parameter);

            int[] funcShape = F.Shape;
            int[] paramShape = Parameter.Shape;

            List<int> fullShape = new();
            fullShape.AddRange(funcShape);
            fullShape.AddRange(paramShape);

            Tensor<T> result = new Tensor<T>(fullShape.ToArray(), funcShape, paramShape);

            void RecurseParams(List<int> paramIndex, int depth)
            {
                if (depth == paramShape.Length)
                {
                    // 미분할 위치에서 파라미터 한 칸 이동
                    Tensor<T> paramShifted = Parameter.Clone();
                    dynamic oldVal = paramShifted.Get(paramIndex.ToArray());
                    paramShifted.Set(paramIndex.ToArray(), oldVal + Epsilon);

                    Tensor<T> fShifted = Function(paramShifted);
                    Tensor<T> fOriginal = F;

                    // ∂f/∂x ≈ (f(x+ε) - f(x)) / ε
                    Tensor<T> diff = (fShifted - fOriginal) / Epsilon;

                    // 결과를 result에 다 채워넣기
                    void RecurseOutput(List<int> outIndex, int d)
                    {
                        if (d == funcShape.Length)
                        {
                            List<int> jacIndex = new(outIndex);
                            jacIndex.AddRange(paramIndex);
                            result.Set(jacIndex.ToArray(), diff.Get(outIndex.ToArray()));
                            return;
                        }

                        for (int i = 0; i < funcShape[d]; i++)
                        {
                            outIndex.Add(i);
                            RecurseOutput(outIndex, d + 1);
                            outIndex.RemoveAt(outIndex.Count - 1);
                        }
                    }

                    RecurseOutput(new List<int>(), 0);
                }
                else
                {
                    for (int i = 0; i < paramShape[depth]; i++)
                    {
                        paramIndex.Add(i);
                        RecurseParams(paramIndex, depth + 1);
                        paramIndex.RemoveAt(paramIndex.Count - 1);
                    }
                }
            }

            RecurseParams(new List<int>(), 0);
            return result;
        }
        */

        
        //public static Tensor<T> Jacobian(Func<Dictionary<string,Tensor<T>>,Tensor<T>> Function, Dictionary<string,Tensor<T>> Parameter)
        public static Tensor<T> Jacobian(Func<Tensor<T>,Tensor<T>> Function, Tensor<T> Parameter, T Epsilon)
        {
            List<int> OutShape = new();

            Tensor<T> F = Function(Parameter);

            for (int i = 0; i < F.Rank; i++)
                OutShape.Add(F.Shape[i]);
            for (int i = 0; i < Parameter.Rank; i++)
                OutShape.Add(Parameter.Shape[i]);

            Tensor<T> OneParameterMoveGetFunctionTensor = F.Clone();
            Tensor<T> result = new(OutShape.ToArray(), (int[])F.Shape.Clone(), (int[])Parameter.Shape.Clone());

            Tensor<T> PartialDifferential(Func<Tensor<T>, Tensor<T>> Function, Tensor<T> Params, Tensor<T> Params_OneMoveP, Tensor<T> Params_OneMoveN)
            {
                return (Function(Params_OneMoveP) - Function(Params_OneMoveN)) / (2 * (dynamic)Epsilon);
            }

            void Recurse_JacobianSetF(List<int> NowIndex, int depth, int[] Index)
            {
                if (depth == F.Rank)
                {
                    List<int> JacobianIndex = new(NowIndex);
                    JacobianIndex.AddRange(Index);
                    result.Set(JacobianIndex.ToArray(),
                        OneParameterMoveGetFunctionTensor.Get(NowIndex.ToArray()));

                    //Tensor<T> Parameter_OneMove = Parameter.Clone();
                    //Parameter_OneMove.Set(
                    //    NowIndex.ToArray(),
                    //    (dynamic)Parameter_OneMove.Get(NowIndex.ToArray()) + Epsilon
                    //    );
                    //result.Set(JacobianIndex.ToArray(),
                    //    PartialDifferential(Function,Parameter, Parameter_OneMove).Get(NowIndex.ToArray()));
                    //Console.WriteLine(string.Join(",", NowIndex));
                }
                else
                {
                    for (int i = 0; i < F.Shape[depth]; i++)
                    {
                        NowIndex.Add(i);
                        Recurse_JacobianSetF(NowIndex, depth + 1, Index);
                        NowIndex.RemoveAt(NowIndex.Count - 1);
                    }
                }
            }

            void Recurse_ParamsAccess(List<int> NowIndex, int depth)
            {
                if (depth == Parameter.Rank)
                {
                    Tensor<T> Parameter_OneMoveP = Parameter.Clone();
                    Tensor<T> Parameter_OneMoveN = Parameter.Clone();
                    Parameter_OneMoveP.Set(
                        NowIndex.ToArray(),
                        (dynamic)Parameter_OneMoveP.Get(NowIndex.ToArray()) + Epsilon
                        );
                    Parameter_OneMoveN.Set(
                        NowIndex.ToArray(),
                        (dynamic)Parameter_OneMoveN.Get(NowIndex.ToArray()) - Epsilon
                        );

                    OneParameterMoveGetFunctionTensor = PartialDifferential(Function, Parameter, Parameter_OneMoveP, Parameter_OneMoveN);

                    Recurse_JacobianSetF(new List<int>(), 0, NowIndex.ToArray());
                }
                else
                {
                    for (int i = 0; i < Parameter.Shape[depth]; i++)
                    {
                        NowIndex.Add(i);
                        Recurse_ParamsAccess(NowIndex, depth + 1);
                        NowIndex.RemoveAt(NowIndex.Count - 1);
                    }
                }
            }

            Recurse_ParamsAccess(new List<int>(), 0);
            return result;
        }
        

        public static Tensor<T> TensorDot(Tensor<T> A, Tensor<T> B, int[] axisA, int[] axisB)
        {
            bool IsAScalar = true;
            bool IsBScalar = true;
            for (int i = 0; i < A.Rank; i++)
                if (A.Shape[i] != 1)
                    IsAScalar = false;
            for (int i = 0; i < B.Rank; i++)
                if (B.Shape[i] != 1)
                    IsBScalar = false;

            if(IsAScalar || IsBScalar)
            {
                int[] AIndex = (int[])A.Shape.Clone();
                int[] BIndex = (int[])B.Shape.Clone();
                for (int i = 0; i < AIndex.Length; i++)
                {
                    AIndex[i] = 0;
                }
                for (int i = 0; i < BIndex.Length; i++)
                {
                    BIndex[i] = 0;
                }

                if (IsAScalar && !IsBScalar)
                    return (dynamic)A.ToScalar() * B;
                if (!IsAScalar && IsBScalar)
                    return (dynamic)B.ToScalar() * A;
                if (IsAScalar && IsBScalar)
                {
                    int[] Shape;
                    if (A.Rank > B.Rank)
                        Shape = (int[])A.Shape.Clone();
                    else
                        Shape = (int[])B.Shape.Clone();

                    Tensor<T> Out = new(Shape);

                    for(int i = 0; i < Shape.Length; i++)
                    {
                        Shape[i] = 0;
                    }

                    Out.Set(Shape, (dynamic)A.Get(AIndex) * (dynamic)B.Get(BIndex));
                    return Out;
                }
            }

            if (axisA.Length != axisB.Length)
                throw new ArgumentException("Axis Lengths must macth");

            for (int i = 0; i < axisA.Length; i++)
                if (A.Shape[axisA[i]] != B.Shape[axisB[i]])
                    throw new ArgumentException("Axis Dimensions do not match");

            List<int> OutShape = new();
            List<int> mapA = new();
            List<int> mapB = new();

            for (int i = 0; i < A.Rank; i++)
                if (!axisA.Contains(i))
                { 
                    OutShape.Add(A.Shape[i]);
                    mapA.Add(i); 
                }
            for (int i = 0; i < B.Rank; i++)
                if (!axisB.Contains(i))
                {
                    OutShape.Add(B.Shape[i]);
                    mapB.Add(i);
                }

            Tensor<T> result = new(OutShape.ToArray());

            void Recurse(List<int> outputIndex, int depth)
            {
                if (depth == OutShape.Count)
                {
                    dynamic sum = 0;
                    int sumDim = A.Shape[axisA[0]];
                    for (int k = 0; k < sumDim; k++)
                    {
                        int[] aIndex = new int[A.Rank];
                        int[] bIndex = new int[B.Rank];

                        int idx = 0;
                        foreach (var i in Enumerable.Range(0, A.Rank))
                        {
                            if (axisA.Contains(i)) aIndex[i] = k;
                            else aIndex[i] = outputIndex[idx++];
                        }

                        idx = mapA.Count;
                        foreach (var i in Enumerable.Range(0, B.Rank))
                        {
                            if (axisB.Contains(i)) bIndex[i] = k;
                            else bIndex[i] = outputIndex[idx++];
                        }

                        sum += (dynamic)A.Get(aIndex) * (dynamic)B.Get(bIndex);
                    }
                    result.Set(outputIndex.ToArray(), sum);
                    return;
                }

                for (int i = 0; i < OutShape[depth]; i++)
                {
                    outputIndex.Add(i);
                    Recurse(outputIndex, depth + 1);
                    outputIndex.RemoveAt(outputIndex.Count - 1);
                }
            }

            Recurse(new List<int>(), 0);
            return result;
        }

        public static Tensor<T> MatrixDot(Tensor<T> A, Tensor<T> B)
        {
            if (A.Rank != 2 || B.Rank != 2)
                throw new Exception();

            return TensorDot(A, B, new int[] { 1 }, new int[] { 0 });
        }

        /*
        public struct TensorInnerState
        {
            public Tensor<T> A;
            public Tensor<T> B;
            public Tensor<T> Out;
            public int[] AxisA;
            public int[] AxisB;

            public TensorInnerState(Tensor<T> a, Tensor<T> b, Tensor<T> o, int[] axA, int[] axB)
            {
                A = a;
                B = b;
                Out = o;
                AxisA = axA;
                AxisB = axB;
            }
        }

        public static int[] GetIndex(int[] outerIndex, int[] axis, int[] innerIndex, int rank)
        {
            List<int> res = new();
            int o = 0, a = 0;
            for (int i = 0; i < rank; i++)
            {
                if (axis.Contains(i)) res.Add(innerIndex[a++]);
                else res.Add(outerIndex[o++]);
            }
            return res.ToArray();
        }

        public static DeepForHelperEnum InnerLoop(DeepForStruct dfs, TensorInnerState state)
        {
            int[] outer = state.Out.Shape.Select((_, i) => dfs.NowIndex[i]).ToArray();
            int sumDim = state.A.Shape[state.AxisA[0]];
            dynamic sum = 0;

            for (int k = 0; k < sumDim; k++)
            {
                int[] inner = new int[] { k };
                var aIdx = GetIndex(outer, state.AxisA, inner, state.A.Rank);
                var bIdx = GetIndex(outer, state.AxisB, inner, state.B.Rank);

                sum += (dynamic)state.A.Get(aIdx) * (dynamic)state.B.Get(bIdx);
            }

            state.Out.Set(outer, sum);
            return DeepForHelperEnum.Return;
        }

        public static Tensor<T> TensorInnerProduct(Tensor<T> A, Tensor<T> B, int[] axisA, int[] axisB)
        {
            if (axisA.Length != axisB.Length)
                throw new ArgumentException("Axis lengths must match");

            for (int i = 0; i < axisA.Length; i++)
                if (A.Shape[axisA[i]] != B.Shape[axisB[i]])
                    throw new ArgumentException("Shapes do not align for inner product");

            var outShape = new List<int>();
            for (int i = 0; i < A.Rank; i++)
                if (!axisA.Contains(i)) outShape.Add(A.Shape[i]);
            for (int i = 0; i < B.Rank; i++)
                if (!axisB.Contains(i)) outShape.Add(B.Shape[i]);

            Tensor<T> Out = new(outShape.ToArray());
            var state = new TensorInnerState(A, B, Out, axisA, axisB);
            var forHelper = new DeepForHelper<TensorInnerState>(InnerLoop, state, Out.Shape);
            forHelper.Run();
            return Out;
        }
        */

        /**
        public static int[] GetComplexIndex(int[] ThisTensorIndex, int[] axis, int[] InnerForIndex, int ThisTensorRank)
        {
            List<int> Out = new();

            int axisPointer = 0;
            int ThisPointer = 0;

            HashSet<int> axisSet = new(axis);

            for(int i = 0; i < ThisTensorRank; i++)
            {
                
                if (!axisSet.Contains(i))
                {
                    Out.Add(InnerForIndex[axisPointer]);
                    axisPointer++;
                }
                else
                {
                    Out.Add(ThisTensorIndex[ThisPointer]);
                    ThisPointer++;
                }
                
            }
            System.Console.WriteLine($"axis {axis[0]}");
            System.Console.WriteLine($"ThisTensorIndex {ThisTensorIndex[0]}");
            System.Console.WriteLine($"InnerForIndoex {InnerForIndex[0]} {InnerForIndex[1]}");
            System.Console.WriteLine($"Out {Out[0]} {Out[1]}");

            return Out.ToArray();
        }
        
        public static DeepForHelperEnum InnerTensorFor(DeepForStruct DFs, TwoTensor<T> IOTensor)
        {
            TwoTensor<T> IOIO = new(
                IOTensor.Tensor1,
                IOTensor.Tensor2,
                IOTensor.OutTensor,
                IOTensor.axis1,
                IOTensor.axis2,
                DFs.NowIndex.ToArray(),
                IOTensor.OutNotIndexA,
                IOTensor.OutNotIndexB
            );

            DeepForHelper<TwoTensor<T>> ForSum = new(InnerTensorInSumFor, IOIO, IOTensor.NowIndex);
            ForSum.Run();

            return DeepForHelperEnum.Return;
        }
        public static DeepForHelperEnum InnerTensorInSumFor(DeepForStruct DFs, TwoTensor<T> IOTensor)
        {
            System.Console.WriteLine($"{IOTensor.NowIndex[0]} {IOTensor.NowIndex[1]}--------------------");

            var A = (dynamic)IOTensor.Tensor1.Get(GetComplexIndex(DFs.NowIndex.ToArray(), IOTensor.axis1, IOTensor.NowIndex, IOTensor.Tensor1.Rank));
            var B = (dynamic)IOTensor.Tensor2.Get(GetComplexIndex(DFs.NowIndex.ToArray(), IOTensor.axis2, IOTensor.NowIndex, IOTensor.Tensor2.Rank));

            //var A = (dynamic)IOTensor.Tensor1.Get(GetComplexIndex(IOTensor.OutNotIndexA, IOTensor.axis1, IOTensor.NowIndex, IOTensor.Tensor1.Rank));
            //var B = (dynamic)IOTensor.Tensor2.Get(GetComplexIndex(IOTensor.OutNotIndexB, IOTensor.axis2, IOTensor.NowIndex, IOTensor.Tensor2.Rank));

            IOTensor.OutTensor.Set(IOTensor.NowIndex,
                (dynamic)IOTensor.OutTensor.Get(IOTensor.NowIndex) + A * B);
            return DeepForHelperEnum.Return;
        }

        
        public static Tensor<T> TensorInnerProduct(Tensor<T> A, Tensor<T> B, int[] axisA, int[] axisB)
        {
            //새로운 공간에 대한 인스턴트 참조 끌고가면서
            //합칠 축에 대한 int[](axis)를 제외한 인덱스 길이(index)를 늘리면서 깊이를 늘리고
            //깊이가 새로운 공간보다 더 늘어나면 멈추고
            //새로운 공간에 대한 인스턴트 참조에 Set 한다
            //이때 Set은 또 다른 for덩어리에서 이뤄지는데
            //axis에 대해서 똑같은 방법으로 접근(axis의 dim만큼 for 중첩 및 axis의 [n]의 값 만큼 for 실행)
            //하여 Set(Out.Get()+A_index*B_index)로 한다

            if (axisA.Length != axisB.Length)
                return null;

            List<int> OutShape = new();

            List<int> AxisShape = new();

            List<int> OutNotIndexA = new();
            List<int> OutNotIndexB = new();

            for (int i = 0; i < axisA.Length; i++)
            {
                if (A.Shape[axisA[i]] != B.Shape[axisB[i]])
                    return null;
                else
                    AxisShape.Add(A.Shape[axisA[i]]);
            }

            for (int i = 0; i < A.Rank; i++)
            {
                if (!axisA.Contains(i))
                {
                    OutShape.Add(A.Shape[i]);
                }
                else
                {
                    OutNotIndexA.Add(A.Shape[i]);
                }
            }
            for (int i = 0; i < B.Rank; i++)
            {
                if (!axisB.Contains(i))
                {
                    OutShape.Add(B.Shape[i]);
                }
                else
                {
                    OutNotIndexA.Add(B.Shape[i]);
                }
            }

            Tensor<T> Out = new(OutShape.ToArray());

            TwoTensor<T> IO = new(A, B, Out,axisA, axisB, AxisShape.ToArray(),
                OutNotIndexA.ToArray(), OutNotIndexB.ToArray());

            DeepForHelper<TwoTensor<T>> For = new(InnerTensorFor, IO, OutShape.ToArray());
            For.Run();

            return Out;
        }
        **/

        /*
        public static Tensor<T> TensorInnerProduct(Tensor<T> A, Tensor<T> B, int[] axisA, int[] axisB)
        {
            if (axisA.Length != axisB.Length)
                return null;

            // shape 검증
            for (int i = 0; i < axisA.Length; i++)
            {
                if (A.Shape[axisA[i]] != B.Shape[axisB[i]])
                    return null;
            }

            // 출력 shape 계산
            List<int> OutShape = new();

            List<int> axisShape = new();
            for (int i = 0; i < A.Rank; i++)
            {
                if (!axisA.Contains(i))
                    OutShape.Add(A.Shape[i]);
            }
            for (int i = 0; i < B.Rank; i++)
            {
                if (!axisB.Contains(i))
                    OutShape.Add(B.Shape[i]);
            }

            // 결과 텐서 생성
            Tensor<T> Out = new(OutShape.ToArray());

            // 중첩 for: Out 정의역 순회
            int[] outShape = Out.Shape;
            int outRank = outShape.Length;

            int[] outIndex = new int[outRank];

            void OuterRecurse(int depth)
            {
                if (depth == outRank)
                {
                    // outerIndex = outIndex
                    // AIndex, BIndex 만들어서 sum over 축

                    dynamic sum = 0;

                    int axisDim = A.Shape[axisA[0]]; // 축 수 (동일한 걸로 가정)

                    for (int k = 0; k < axisDim; k++)
                    {
                        int[] aIndex = new int[A.Rank];
                        int[] bIndex = new int[B.Rank];

                        // A index 구성
                        int ai = 0;
                        for (int i = 0; i < A.Rank; i++)
                        {
                            if (axisA.Contains(i))
                                aIndex[i] = k;
                            else
                                aIndex[i] = outIndex[ai++];
                        }

                        // B index 구성
                        int bi = A.Rank - axisA.Length;
                        for (int i = 0; i < B.Rank; i++)
                        {
                            if (axisB.Contains(i))
                                bIndex[i] = k;
                            else
                                bIndex[i] = outIndex[bi++];
                        }

                        sum += (dynamic)A.Get(aIndex) * (dynamic)B.Get(bIndex);
                    }

                    Out.Set(outIndex, sum);
                    return;
                }

                for (int i = 0; i < outShape[depth]; i++)
                {
                    outIndex[depth] = i;
                    OuterRecurse(depth + 1);
                }
            }

            OuterRecurse(0);

            return Out;
        }
        */


    }
}

