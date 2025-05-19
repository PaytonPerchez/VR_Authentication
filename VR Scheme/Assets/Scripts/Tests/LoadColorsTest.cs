using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEngine.UI;

public class LoadColorsTest : MonoBehaviour
{
    [SerializeField] private string path;
    [SerializeField] private Image displayedImage;
    
    private Color[] colors;
    private float startTime;
    private int currentIndex;

    // Start is called before the first frame update
    void Start()
    {
        colors = LoadColors(path);
        startTime = Time.time;
        currentIndex = 0;
    }

    // Update is called once per frame
    void Update()
    {
        if((Time.time - startTime) >= 1)
        {
            if (currentIndex == (colors.Length - 1))
            {
                currentIndex = 0;
            }
            else
            {
                currentIndex++;
            }
            startTime = Time.time;
            displayedImage.color = colors[currentIndex];
        }
    }

    private Color[] LoadColors(string path)
    {
        try
        {
            using StreamReader reader = new(path);

            string[] lines = reader.ReadToEnd().Split(new char[] { '\n' });
            int linesPerColor = 5;

            Color[] colors = new Color[lines.Length / linesPerColor];
            float r, g, b, a;

            for (int i = 0; i < lines.Length; i += linesPerColor)
            {
                Debug.Log("R: " + lines[i] + ", G: " + lines[i + 1] + ", B: " + lines[i + 2] + ", A: " + lines[i + 3]);
                r = float.Parse(lines[i]);
                g = float.Parse(lines[i + 1]);
                b = float.Parse(lines[i + 2]);
                a = float.Parse(lines[i + 3]);

                Debug.Log("R: " + r + ", G: " + g + ", B: " + b + ", A: " + a);
                colors[i / linesPerColor] = new(r, g, b, a);
            }

            return colors;
        }
        catch (IOException e)
        {
            Debug.Log("The file could not be read...");
            Debug.Log(e.Message);

            return new Color[0];
        }
    }

    private void PrintArray(Color[] array)
    {
        string elements = "";
        for (int i = 0; i < array.Length; i++)
        {
            elements += array[i] + ", ";
        }
        Debug.Log(elements);
    }
}
