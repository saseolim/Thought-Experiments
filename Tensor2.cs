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

/*
 int[]방식으로 tensor구현하기로 결정
역전파 구현 예정
 */

namespace Jacobian_Test
{
    public class Tensor2<T>
    {
        public Array tensor { get; private set; }
        public int Rank { get => Shape.Length ; }
        public int[] Shape { get; private set; }
        public bool IsJacobian { get; private set; }
        public int[] JacobianFuncShape { get; private set; }
        public int JF_Rank { get => JacobianFuncShape.Length; }
        public int[] JacobianParamsShape { get; private set; }
        public int JP_Rank { get => JacobianParamsShape.Length; }

        public Tensor2(int[] Shape)
        {
            this.Shape = (int[])Shape.Clone();
            this.tensor = Array.CreateInstance(typeof(T), (int[])Shape.Clone());
            //딕셔너리 Dictionary<int[],T>고러바람

            this.IsJacobian = false;
            this.JacobianFuncShape = new int[0];
            this.JacobianParamsShape = new int[0];
        }
        public Tensor2(int[] Shape, int[] JacobianFuncShape, int[] JacobianParamsShape)
        {
            this.Shape = Shape;
            this.tensor = Array.CreateInstance(typeof(T), Shape);

            this.IsJacobian = true;
            this.JacobianFuncShape = (int[])JacobianFuncShape.Clone();
            this.JacobianParamsShape = (int[])JacobianParamsShape.Clone();
        }
        public Tensor2(Tensor2<T> OneBone)
        {
            this.tensor = (Array)OneBone.tensor.Clone();
            this.Shape = (int[])OneBone.Shape.Clone();

            this.IsJacobian = OneBone.IsJacobian;
            this.JacobianFuncShape = (int[])OneBone.JacobianFuncShape.Clone();
            this.JacobianParamsShape = (int[])OneBone.JacobianParamsShape.Clone();
        }
        public Tensor2(Tensor2<T> OneBone, int[] JacobianFuncShape, int[] JacobianParamsShape)
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

        public Tensor2<T> Clone()
        {
            Tensor2<T> clone = new Tensor2<T>(this.Shape);
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

        public Tensor2<T> Transpose2D()
        {
            var shape = new int[] { this.Shape[1], this.Shape[0] };
            var result = new Tensor2<T>(shape);

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

        public static Tensor2<T> TensorOneOperation(Tensor2<T> tensor,
            Func<T,T> function)
        {
            Tensor2<T> Out = new(tensor.Shape);

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

        public static Tensor2<T> TensorTwoOperation(Tensor2<T> tensor1 , Tensor2<T> tensor2,
            Func<T,T,T> function)
        {
            if (tensor1.Rank != tensor2.Rank)
                throw new ArgumentException("두 Tensor의 Rank가 일치하지 않습니다.");
            for(int i = 0; i < tensor1.Rank; i++)
            {
                if (tensor1.Shape[i] != tensor2.Shape[i])
                    throw new ArgumentException("두 Tensor의 Shape가 일치하지 않습니다.");
            }
            Tensor2<T> Out = new(tensor1.Shape);

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

        public static Tensor2<T> TensorTwoOperation(Tensor2<T> tensor, T K,
            Func<T, T, T> function)
        {
            Tensor2<T> Out = new(tensor.Shape);

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

        public static Tensor2<T> operator +(Tensor2<T> A, Tensor2<T> B)
        {
            return TensorTwoOperation(A, B,
                (a, b) =>
                {
                    return (dynamic)a + (dynamic)b;
                });
        }
        public static Tensor2<T> operator +(Tensor2<T> A, T K)
        {
            return TensorTwoOperation(A, K,
                (a, k) =>
                {
                    return (dynamic)a + (dynamic)k;
                });
        }
        public static Tensor2<T> operator +(T K, Tensor2<T> A)
        {
            return TensorTwoOperation(A, K,
                (a, k) =>
                {
                    return (dynamic)a + (dynamic)k;
                });
        }

        public static Tensor2<T> operator -(Tensor2<T> A, Tensor2<T> B)
        {
            return TensorTwoOperation(A, B,
                (a,b) =>
                {
                     return (dynamic)a - (dynamic)b;
                });
        }
        public static Tensor2<T> operator -(Tensor2<T> A, T K)
        {
            return TensorTwoOperation(A, K,
                (a, k) =>
                {
                    return (dynamic)a - (dynamic)k;
                });
        }
        public static Tensor2<T> operator -(T K, Tensor2<T> A)
        {
            return TensorTwoOperation(A, K,
                (a, k) =>
                {
                    return (dynamic)a - (dynamic)k;
                });
        }

        public static Tensor2<T> operator *(Tensor2<T> A, Tensor2<T> B)
        {
            return TensorTwoOperation(A, B,
                (a, b) =>
                {
                    return (dynamic)a * (dynamic)b;
                });
        }

        public static Tensor2<T> operator *(Tensor2<T> A, T K)
        {
            return TensorTwoOperation(A, K,
                (a, k) =>
                {
                    return (dynamic)a * (dynamic)k;
                });
        }

        public static Tensor2<T> operator *(T K, Tensor2<T> A)
        {
            return TensorTwoOperation(A, K,
                (a, k) =>
                {
                    return (dynamic)a * (dynamic)k;
                });
        }

        public static Tensor2<T> operator /(Tensor2<T> A, T K)
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

        public static decimal FrobeniusNorm(Tensor2<T> tensor)
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
        public static Tensor2<T> Chain(Tensor2<T> InputParamter, T Epsilon, params Func<Tensor2<T>,Tensor2<T>>[] Functions) 
        {
            List<List<int>> AllForShapes = new();
            AllForShapes.Add(new List<int>(InputParamter.Shape));
            List<Tensor2<T>> Tensors = new();
            Tensors.Add(InputParamter);
            for (int i = Functions.Length - 1; i > -1; i--)
            {
                Tensor2<T> ThisShapeInput = Tensors[Tensors.Count - 1];
                ThisShapeInput = Functions[i](ThisShapeInput);
                Tensors.Add(ThisShapeInput);
                AllForShapes.Add(new List<int>(ThisShapeInput.Shape));
            }
            AllForShapes.Reverse();
            Tensors.Reverse();

            List<Tensor2<T>> Jacobians = new();
            Tensor2<T> JacobianMemory = Tensors[Tensors.Count - 1];
            for (int i = Functions.Length - 1; i > -1; i--)
            {
                Jacobians.Add(Jacobian(Functions[i], JacobianMemory, Epsilon));
                JacobianMemory = Tensors[i];
            }
            Jacobians.Reverse();

            Tensor2<T> TensorDotMemonry = Jacobians[0];

            
            for (int i = 1; i < Jacobians.Count; i++)
            {

                //정확한 축 매칭 (출력축 = 앞 텐서 func, 입력축 = 뒷 텐서 param)
                int JFM_A = TensorDotMemonry.JF_Rank;
                int JPP_B = Jacobians[i].JP_Rank;

                int[] axisA = Enumerable.Range(TensorDotMemonry.JF_Rank, TensorDotMemonry.JP_Rank).ToArray(); // 앞: 입력 축
                int[] axisB = Enumerable.Range(0, Jacobians[i].JF_Rank).ToArray(); // 뒤: 출력 축


                TensorDotMemonry = new Tensor2<T>(TensorDot(TensorDotMemonry, Jacobians[i], axisA.ToArray(), axisB.ToArray()), TensorDotMemonry.JacobianFuncShape, Jacobians[i].JacobianParamsShape);
            }

            return TensorDotMemonry;
        }

        public static Tensor2<T> Jacobian(Func<Tensor2<T>, Dictionary<string,Tensor2<T>>, Tensor2<T>> Function, Tensor2<T> Parameter, Dictionary<string,Tensor2<T>> StaticParmeter,
    T Epsilon)
        {
            List<int> OutShape = new();

            Tensor2<T> F = Function(Parameter, StaticParmeter);

            for (int i = 0; i < F.Rank; i++)
                OutShape.Add(F.Shape[i]);
            for (int i = 0; i < Parameter.Rank; i++)
                OutShape.Add(Parameter.Shape[i]);

            Tensor2<T> OneParameterMoveGetFunctionTensor = F.Clone();
            Tensor2<T> result = new(OutShape.ToArray(), (int[])F.Shape.Clone(), (int[])Parameter.Shape.Clone());

            Tensor2<T> PartialDifferential(Func<Tensor2<T>, Dictionary<string,Tensor2<T>>, Tensor2<T>> Function, Tensor2<T> Params, Tensor2<T> Params_OneMoveP, Tensor2<T> Params_OneMoveN)
            {
                return (Function(Params_OneMoveP, StaticParmeter) - Function(Params_OneMoveN, StaticParmeter)) / (2 * (dynamic)Epsilon);
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
                    Tensor2<T> Parameter_OneMoveP = Parameter.Clone();
                    Tensor2<T> Parameter_OneMoveN = Parameter.Clone();
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

        public static Tensor2<T> Jacobian(Func<Tensor2<T>, List<Tensor2<T>>, Tensor2<T>> Function, Tensor2<T> Parameter, List<Tensor2<T>> StaticParmeter, T Epsilon)
        {
            List<int> OutShape = new();

            Tensor2<T> F = Function(Parameter, StaticParmeter);

            for (int i = 0; i < F.Rank; i++)
                OutShape.Add(F.Shape[i]);
            for (int i = 0; i < Parameter.Rank; i++)
                OutShape.Add(Parameter.Shape[i]);

            Tensor2<T> OneParameterMoveGetFunctionTensor = F.Clone();
            Tensor2<T> result = new(OutShape.ToArray(), (int[])F.Shape.Clone(), (int[])Parameter.Shape.Clone());

            Tensor2<T> PartialDifferential(Func<Tensor2<T>, List<Tensor2<T>>, Tensor2<T>> Function, Tensor2<T> Params, Tensor2<T> Params_OneMoveP, Tensor2<T> Params_OneMoveN)
            {
                return (Function(Params_OneMoveP,StaticParmeter) - Function(Params_OneMoveN,StaticParmeter)) / (2 * (dynamic)Epsilon);
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
                    Tensor2<T> Parameter_OneMoveP = Parameter.Clone();
                    Tensor2<T> Parameter_OneMoveN = Parameter.Clone();
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

        public static Tensor2<T> Jacobian(Func<Tensor2<T>,Tensor2<T>> Function, Tensor2<T> Parameter, T Epsilon)
        {
            List<int> OutShape = new();

            Tensor2<T> F = Function(Parameter);

            for (int i = 0; i < F.Rank; i++)
                OutShape.Add(F.Shape[i]);
            for (int i = 0; i < Parameter.Rank; i++)
                OutShape.Add(Parameter.Shape[i]);

            Tensor2<T> OneParameterMoveGetFunctionTensor = F.Clone();
            Tensor2<T> result = new(OutShape.ToArray(), (int[])F.Shape.Clone(), (int[])Parameter.Shape.Clone());

            Tensor2<T> PartialDifferential(Func<Tensor2<T>, Tensor2<T>> Function, Tensor2<T> Params, Tensor2<T> Params_OneMoveP, Tensor2<T> Params_OneMoveN)
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
                    Tensor2<T> Parameter_OneMoveP = Parameter.Clone();
                    Tensor2<T> Parameter_OneMoveN = Parameter.Clone();
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
        

        public static Tensor2<T> TensorDot(Tensor2<T> A, Tensor2<T> B, int[] axisA, int[] axisB)
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

                    Tensor2<T> Out = new(Shape);

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

            Tensor2<T> result = new(OutShape.ToArray());

            //LLM에게 입력한 파라미터
            //C#로 int[]를 넣으면 만약 {2, 3}를 넣으면 {{0,0}, {0, 1}, {0, 2}, {1,0}, {1,1}, {1,2}}인 int[][]가 나오게 하는 함수를 만들어줘
            int[][] GenerateIndices(int[] dimensions) //LLM도움 받음
            {
                if (dimensions == null || dimensions.Length == 0)
                {
                    return new int[0][];
                }

                // 재귀 함수 호출을 위한 헬퍼 리스트
                List<int[]> resultList = new List<int[]>();
                int[] currentIndices = new int[dimensions.Length]; // 현재 조합되는 인덱스 배열

                // 인덱스 0부터 시작하는 재귀 호출
                GenerateIndicesRecursive(dimensions, 0, currentIndices, resultList);

                return resultList.ToArray();
            }

            void GenerateIndicesRecursive(int[] dimensions, int dimensionIndex, int[] currentIndices, List<int[]> resultList) //LLM도움 받음
            {
                // 종료 조건: 모든 차원의 인덱스를 설정한 경우
                if (dimensionIndex == dimensions.Length)
                {
                    // 현재 인덱스 조합을 결과 리스트에 추가합니다.
                    // *중요*: currentIndices를 그대로 추가하면 나중에 값이 변경될 수 있으므로,
                    // 새로운 배열에 복사하여 추가해야 합니다.
                    resultList.Add((int[])currentIndices.Clone());
                    return;
                }

                // 현재 차원(dimensionIndex)의 크기만큼 반복합니다.
                int limit = dimensions[dimensionIndex];
                for (int i = 0; i < limit; i++)
                {
                    currentIndices[dimensionIndex] = i; // 현재 차원의 인덱스 설정

                    // 다음 차원으로 재귀 호출
                    GenerateIndicesRecursive(dimensions, dimensionIndex + 1, currentIndices, resultList);
                }
            }

            void Recurse(List<int> outputIndex, int depth)
            {
                if (depth == OutShape.Count)
                {
                    dynamic sum = 0;
                    int sumDim = 1;
                    int[] sumDimArray = new int[axisA.Length];
                    for (int i =0; i < axisA.Length; i++)
                    {
                        sumDim *= A.Shape[axisA[i]];
                        sumDimArray[i] = A.Shape[axisA[i]];
                    }

                    int[][] Sum2DArray = GenerateIndices(sumDimArray);

                    for (int k = 0; k < sumDim; k++)
                    {
                        int[] aIndex = new int[A.Rank];
                        int[] bIndex = new int[B.Rank];

                        int idx = 0;
                        int ido = 0;
                        foreach (var i in Enumerable.Range(0, A.Rank))
                        {
                            if (axisA.Contains(i)) //aIndex[i] = k;
                            {
                                aIndex[i] = Sum2DArray[k][ido++];
                            }
                            else aIndex[i] = outputIndex[idx++];
                        }

                        idx = mapA.Count;
                        ido = 0;
                        foreach (var i in Enumerable.Range(0, B.Rank))
                        {
                            if (axisB.Contains(i)) //bIndex[i] = k;
                            {
                                bIndex[i] = Sum2DArray[k][ido++];
                            }
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

        public static Tensor2<T> MatrixDot(Tensor2<T> A, Tensor2<T> B)
        {
            if (A.Rank != 2 || B.Rank != 2)
                throw new Exception();

            return TensorDot(A, B, new int[] { 1 }, new int[] { 0 });
        }
    }
}

