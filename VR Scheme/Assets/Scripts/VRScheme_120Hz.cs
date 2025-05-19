using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using VIVE.OpenXR;
using VIVE.OpenXR.EyeTracker;
using System.IO;
using System;
using System.Threading;
using TMPro;

public class VRScheme_120Hz : MonoBehaviour
{
    [SerializeField] private string userPath;
    [SerializeField] private int secsStimShown;
    [SerializeField] private bool useIndexOfLoadedColor;
    [SerializeField] private int colorIndex;
    [SerializeField] private int r;
    [SerializeField] private int g;
    [SerializeField] private int b;
    [SerializeField] private long stimFadeTime;
    [SerializeField] private Image displayedStim;
    [SerializeField] private TMP_Text instructions;
    [SerializeField] private bool useFocus;
    private const long EyeTrackerLoadTime = 7000; // eye tracker takes at least 5 seconds to initialize
    private const int SampleRateHz = 120;

    private static bool isClosing = false; // use to stop runaway recording thread
    private DataRecorder dataRecorder;

    private static bool recordingInProgress = false;
    private bool transitioning; // signifies the stimulus is transitioning between being displayed and being hidden
    private long tempStartTime;
    private long globalStartTime;
    private int leftMissingCount;
    private int rightMissingCount;

    private PupilData[] recordedData; // format == recordedData[originalStimuli.IndexOf(Color)] = List<PupilData>

    // Start is called before the first frame update
    void Start()
    {
        displayedStim.enabled = false;

        if (useIndexOfLoadedColor)
        {
            displayedStim.color = LoadColor(userPath, colorIndex);
            displayedStim.color = new Color(displayedStim.color.r, displayedStim.color.g, displayedStim.color.b, 0);
        }
        else
        {
            displayedStim.color = new Color(r / 255f, g / 255f, b / 255f, 0);
        }

        leftMissingCount = 0;
        rightMissingCount = 0;

        // Set the minimum number of milliseconds of lost data required to restart the recording process for the displayed stimulus
        dataRecorder = new(secsStimShown);

        Debug.Log(secsStimShown * 1000);
        globalStartTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
    }

    // Update is called once per frame
    void Update()
    {
        if (recordingInProgress)
        {
            if (!dataRecorder.IsRecording() && displayedStim.enabled)
            {
                recordedData = dataRecorder.GetData();
                leftMissingCount = dataRecorder.GetMissingLeft();
                rightMissingCount = dataRecorder.GetMissingRight();
                SavePupilData(userPath); // save biometric template

                displayedStim.enabled = false;
                instructions.text = "Recording Complete!";
                instructions.color = Color.white;
                Debug.Log("Done Recording!");
            }
        }
        else if (transitioning)
        {
            if ((DateTimeOffset.Now.ToUnixTimeMilliseconds() - tempStartTime) <= stimFadeTime)
            {
                displayedStim.color = new Color(displayedStim.color.r, displayedStim.color.g, displayedStim.color.b, (DateTimeOffset.Now.ToUnixTimeMilliseconds() - tempStartTime) / (float) stimFadeTime);
            }
            else
            {
                displayedStim.color = new Color(displayedStim.color.r, displayedStim.color.g, displayedStim.color.b, 1);
                transitioning = false;
                recordingInProgress = true;
                Debug.Log("Start");
            }
        }
        else
        {
            XR_HTC_eye_tracker.Interop.GetEyePupilData(out XrSingleEyePupilDataHTC[] out_pupils);
            Debug.Log("(updating)" + ((DateTimeOffset.Now.ToUnixTimeMilliseconds() - globalStartTime) / 1000f) + "s: " + out_pupils[(int)XrEyePositionHTC.XR_EYE_POSITION_LEFT_HTC].pupilDiameter + "mm, " + out_pupils[(int)XrEyePositionHTC.XR_EYE_POSITION_RIGHT_HTC].pupilDiameter + "mm");

            instructions.text = "Starting in " + ((EyeTrackerLoadTime / 1000) - ((DateTimeOffset.Now.ToUnixTimeMilliseconds() - globalStartTime) / 1000)) + " seconds...";

            // Check if the user wants to start the authentication or enrollment process
            if ((DateTimeOffset.Now.ToUnixTimeMilliseconds() - globalStartTime) > EyeTrackerLoadTime)
            {
                instructions.text = useFocus ? "+" : "";
                instructions.color = Color.black;
                transitioning = true;
                displayedStim.enabled = true;
                dataRecorder.StartRecording();
                tempStartTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
            }
        }
    }

    private void OnApplicationQuit()
    {
        isClosing = true;
        dataRecorder.Abort();
    }

    private class PupilData
    {
        readonly long timestampMillis;
        readonly float leftSizeMM;
        readonly float rightSizeMM;

        public PupilData(long timestampMillis, float leftSizeMM, float rightSizeMM)
        {
            this.timestampMillis = timestampMillis;
            this.leftSizeMM = leftSizeMM;
            this.rightSizeMM = rightSizeMM;
        }

        public long GetTimestamp()
        {
            return timestampMillis;
        }

        public string GetData()
        {
            return "(" + timestampMillis + ", " + leftSizeMM + ", " + rightSizeMM + ")";
        }
    }

    private void SavePupilData(string path)
    {
        using (StreamWriter outputFile = new(path + "PupilData" + colorIndex + ".txt"))
        {
            outputFile.WriteLine(displayedStim.color);
            outputFile.WriteLine(new PupilData(-1 * recordedData.Length, leftMissingCount, rightMissingCount).GetData());

            foreach (PupilData data in recordedData)
            {
                outputFile.WriteLine(data.GetData());
            }
        }

        Debug.Log("Recorded sizes: " + recordedData.Length);
        Debug.Log("Original capacity: " + (SampleRateHz * secsStimShown));
        Debug.Log("Saving Complete!");
    }

    private Color LoadColor(string path, int index)
    {
        try
        {
            using StreamReader reader = new(path + "Colors.txt");

            string[] lines = reader.ReadToEnd().Split(new char[] { '\n' });
            int linesPerColor = 5;

            Color[] colors = new Color[lines.Length / linesPerColor];
            float r, g, b, a;

            for (int i = 0; i < lines.Length; i += linesPerColor)
            {
                r = float.Parse(lines[i]);
                g = float.Parse(lines[i + 1]);
                b = float.Parse(lines[i + 2]);
                a = float.Parse(lines[i + 3]);
                colors[i / linesPerColor] = new(r, g, b, a);
                Debug.Log(colors[i / linesPerColor]);
            }

            return colors[index];
        }
        catch (IOException e)
        {
            Debug.Log("The file could not be read...");
            Debug.Log(e.Message);

            return Color.white;
        }
    }

    private class DataRecorder
    {
        private Thread thread;
        private bool isRecording;
        private PupilData[] data;
        private long durationSecs;
        private int leftMissingCount;
        private int rightMissingCount;

        public DataRecorder(int durationSecs)
        {
            this.durationSecs = durationSecs;
            thread = new Thread(RecordData);
            isRecording = false;
            data = new PupilData[0];
            leftMissingCount = 0;
            rightMissingCount = 0;
        }

        public bool StartRecording()
        {
            if (isRecording)
            {
                return false;
            }
            else
            {
                thread = new Thread(RecordData);
                thread.Start();
                return true;
            }
        }

        public bool IsRecording()
        {
            return isRecording;
        }

        public void Abort()
        {
            thread.Abort();
        }

        public PupilData[] GetData()
        {
            // Try to avoid potential threading issues like race conditions (Unity is not thread safe)
            if (isRecording)
            {
                return new PupilData[0];
            }
            else
            {
                return data;
            }
        }

        public int GetMissingRight()
        {
            return rightMissingCount;
        }

        public int GetMissingLeft()
        {
            return leftMissingCount;
        }

        private void RecordData()
        {
            data = new PupilData[SampleRateHz * durationSecs];
            int dataIndex = 0;
            long durationMillis = durationSecs * 1000;
            isRecording = true;

            // Only use these if keeping track of invalid values during and after blinks
            //leftMissingCount = 0;
            //rightMissingCount = 0;

            long startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();

            /*
             * 120Hz = updates every 25/3 = 8.33 milliseconds
             * Some intervals do not divide evenly => 8.33, 16.66, 25
             * 25/3 intervals can be approximated like so: 9, 17, 25
             */
            int[] updateIntervals = new int[] { 9, 8, 8 };
            int updateIndex = 0;

            long currentTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
            long nextUpdateTime = startTime;
            float leftDiameter;
            float rightDiameter;

            do
            {
                if (isClosing)
                {
                    break;
                }

                // Record data at approximately 120Hz
                if (currentTime >= nextUpdateTime)
                {
                    XR_HTC_eye_tracker.Interop.GetEyePupilData(out XrSingleEyePupilDataHTC[] out_pupils);
                    XrSingleEyePupilDataHTC rightPupil = out_pupils[(int)XrEyePositionHTC.XR_EYE_POSITION_RIGHT_HTC];
                    XrSingleEyePupilDataHTC leftPupil = out_pupils[(int)XrEyePositionHTC.XR_EYE_POSITION_LEFT_HTC];

                    leftDiameter = leftPupil.isDiameterValid ? leftPupil.pupilDiameter : -1;

                    rightDiameter = rightPupil.isDiameterValid ? rightPupil.pupilDiameter : -1;

                    // Note: if wanting to avoid data interpolation, only valid diameters should be added
                    data[dataIndex] = new PupilData(currentTime - startTime, leftDiameter, rightDiameter);
                    dataIndex++;

                    // Debug logging seems to be safe
                    Debug.Log((currentTime - startTime) + " | " + "Left: " + leftPupil.pupilDiameter + '(' + leftDiameter + ')' + "mm, Right: " + rightPupil.pupilDiameter + '(' + rightDiameter + ')' + "mm");

                    nextUpdateTime += updateIntervals[updateIndex];
                    updateIndex = (updateIndex + 1) % updateIntervals.Length;
                }

                currentTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
            } while ((currentTime - startTime) < durationMillis);

            isRecording = false;
        }
    }
}
