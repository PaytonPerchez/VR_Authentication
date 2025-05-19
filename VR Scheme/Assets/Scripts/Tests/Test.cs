using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using VIVE.OpenXR;
using VIVE.OpenXR.EyeTracker;
using System;

public class Test : MonoBehaviour
{
    [SerializeField] Image displayedImage;
    [SerializeField] GameObject progressBar;
    [SerializeField] GameObject headset;

    private long startTime;

    // Start is called before the first frame update
    void Start()
    {
        List<int> list = new();
        Debug.Log(list.Count);
        startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
    }

    // Update is called once per frame
    void Update()
    {
        XR_HTC_eye_tracker.Interop.GetEyePupilData(out XrSingleEyePupilDataHTC[] out_pupils);
        XrSingleEyePupilDataHTC leftPupil = out_pupils[(int)XrEyePositionHTC.XR_EYE_POSITION_LEFT_HTC];
        XrSingleEyePupilDataHTC rightPupil = out_pupils[(int)XrEyePositionHTC.XR_EYE_POSITION_RIGHT_HTC];
        Debug.Log(((DateTimeOffset.Now.ToUnixTimeMilliseconds() - startTime) / 1000f) + "s: " + leftPupil.pupilDiameter + "mm, " + rightPupil.pupilDiameter + "mm");
    }
}
