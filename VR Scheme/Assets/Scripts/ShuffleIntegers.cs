using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShuffleIntegers : MonoBehaviour
{
    [SerializeField] private int inclusiveMin;
    [SerializeField] private int exclusiveMax;

    // Start is called before the first frame update
    void Start()
    {
        int[] integers = new int[exclusiveMax - inclusiveMin];
        PrintArray(Shuffle(FillArray(integers, inclusiveMin)));
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private int[] FillArray(int[] array, int min)
    {
        for (int i = 0; i < array.Length; i++)
        {
            array[i] = min + i;
        }

        return array;
    }

    // Source: https://github.com/dotnet/runtime/blob/1d1bf92fcf43aa6981804dc53c5174445069c9e4/src/libraries/System.Private.CoreLib/src/System/Random.cs#L311C13-L324C10
    private int[] Shuffle(int[] array)
    {
        System.Random rand = new();

        for (int i = 0; i < array.Length; i++)
        {
            int j = rand.Next(i, array.Length);

            if (j != i)
            {
                int temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }

        return array;
    }

    private void PrintArray(int[] array)
    {
        string elements = "";
        for (int i = 0; i < array.Length; i++)
        {
            elements += array[i] + ", ";
        }
        Debug.Log(elements.Substring(0, elements.Length - 2));
    }
}
