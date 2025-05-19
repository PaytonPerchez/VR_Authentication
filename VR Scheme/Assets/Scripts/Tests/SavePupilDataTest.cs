using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VIVE.OpenXR;
using VIVE.OpenXR.EyeTracker;
using System.IO;
using System;

public class SavePupilDataTest : MonoBehaviour
{
    [SerializeField] private string path;
    private long startTime;
    private List<PupilData> recordedData;
    private bool saved;

    // Start is called before the first frame update
    void Start()
    {
        saved = false;
        recordedData = new List<PupilData>();
        startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
    }

    // Update is called once per frame
    void Update()
    {
        XR_HTC_eye_tracker.Interop.GetEyePupilData(out XrSingleEyePupilDataHTC[] out_pupils);
        XrSingleEyePupilDataHTC rightPupil = out_pupils[(int)XrEyePositionHTC.XR_EYE_POSITION_RIGHT_HTC];
        XrSingleEyePupilDataHTC leftPupil = out_pupils[(int)XrEyePositionHTC.XR_EYE_POSITION_LEFT_HTC];
        
        // Allow eye tracker 5 seconds to initialize
        if (DateTimeOffset.Now.ToUnixTimeMilliseconds() - startTime >= 5000)
        {
            if (rightPupil.isDiameterValid && leftPupil.isDiameterValid)
            {
                recordedData.Add(new PupilData(DateTimeOffset.Now.ToUnixTimeMilliseconds() - startTime, leftPupil.pupilDiameter, rightPupil.pupilDiameter));
                //Debug.Log(leftPupil.pupilDiameter + ", " + rightPupil.pupilDiameter);
            }
            else
            {
                recordedData.Add(new PupilData(DateTimeOffset.Now.ToUnixTimeMilliseconds() - startTime, -1, -1));
                //Debug.Log(-1 + ", " + -1);
            }

            // Record 4 seconds of pupil size data
            if (((DateTimeOffset.Now.ToUnixTimeMilliseconds() - startTime) >= 9000) && !saved)
            {
                SavePupilData(path);
                saved = true;
            }
        }
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

        public float GetTimestamp()
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
        using (StreamWriter outputFile = new(path))
        {
            foreach(PupilData data in recordedData)
                {
                outputFile.WriteLine(data.GetData());
                //Debug.Log("Progress...");
            }
        }

        Debug.Log("Saving Complete!");
    }
}
