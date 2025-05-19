using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using VIVE.OpenXR;
using VIVE.OpenXR.EyeTracker;

public class EyeOpennessTest : MonoBehaviour
{
    [SerializeField] Image progressBar;
    private float openness;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        openness = EyeOpenness();
        if (openness < 0)
        {
            progressBar.color = Color.red;
            progressBar.fillAmount = 1;
        }
        else
        {
            progressBar.color = Color.green;
            progressBar.fillAmount = openness;
        }
        Debug.Log(openness);
    }

    float EyeOpenness()
    {
        XR_HTC_eye_tracker.Interop.GetEyeGeometricData(out XrSingleEyeGeometricDataHTC[] out_geometrics);
        XrSingleEyeGeometricDataHTC rightGeometric = out_geometrics[(int)XrEyePositionHTC.XR_EYE_POSITION_RIGHT_HTC];
        XrSingleEyeGeometricDataHTC leftGeometric = out_geometrics[(int)XrEyePositionHTC.XR_EYE_POSITION_LEFT_HTC];

        if (rightGeometric.isValid && leftGeometric.isValid)
        {
            return Mathf.Min(leftGeometric.eyeOpenness, rightGeometric.eyeOpenness);
        }
        else
        {
            return -1;
        }
    }
}
