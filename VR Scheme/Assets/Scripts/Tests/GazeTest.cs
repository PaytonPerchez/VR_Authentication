using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using VIVE.OpenXR;
using VIVE.OpenXR.EyeTracker;

public class GazeTest : MonoBehaviour
{
    [SerializeField] Image progressBar;
    [SerializeField] Image displayedStim;
    [SerializeField] GameObject headsetOrigin;
    [SerializeField] GameObject background;
    [SerializeField] GameObject gazeEndPointLeft;
    [SerializeField] GameObject gazeEndPointRight;

    private Vector3 headsetForwardNorm;
    private Vector3 leftGazeDistance;
    private Vector3 rightGazeDistance;

    private Vector3 leftEyePosition;
    private Vector3 rightEyePosition;
    private bool initialized;

    // Start is called before the first frame update
    void Start()
    {
        headsetForwardNorm = (displayedStim.transform.position - headsetOrigin.transform.position).normalized;
        initialized = false;
    }

    // Update is called once per frame
    void Update()
    {
        XR_HTC_eye_tracker.Interop.GetEyeGazeData(out XrSingleEyeGazeDataHTC[] out_gazes);
        XrSingleEyeGazeDataHTC leftGaze = out_gazes[(int)XrEyePositionHTC.XR_EYE_POSITION_LEFT_HTC];
        XrSingleEyeGazeDataHTC rightGaze = out_gazes[(int)XrEyePositionHTC.XR_EYE_POSITION_RIGHT_HTC];

        if (leftGaze.isValid && rightGaze.isValid)
        {
            leftEyePosition = leftGaze.gazePose.position.ToUnityVector();
            rightEyePosition = rightGaze.gazePose.position.ToUnityVector();

            // Get vectors from eyes to stimulus
            if (!initialized)
            {
                leftGazeDistance = (progressBar.transform.position - leftEyePosition).magnitude * headsetForwardNorm;
                rightGazeDistance = (progressBar.transform.position - rightEyePosition).magnitude * headsetForwardNorm;
                
            }

            // Update positions of gaze visualization objects
            gazeEndPointLeft.transform.position = leftGaze.gazePose.orientation.ToUnityQuaternion() * leftGazeDistance + leftEyePosition;
            gazeEndPointRight.transform.position = rightGaze.gazePose.orientation.ToUnityQuaternion() * rightGazeDistance + rightEyePosition;

            // Fill the progress bar if eye gaze vectors align with the vector from the headset to the progress bar according to the specified threshold
            if ((Vector3.Dot(gazeEndPointLeft.transform.position.normalized, progressBar.transform.position.normalized) >= 0.9999))// && (Vector3.Dot(gazeEndPointRight.transform.position.normalized, progressBar.transform.position.normalized) >= 0.9999))
            {
                if (progressBar.fillAmount == 1)
                {
                    progressBar.fillAmount = 0;
                }
                progressBar.fillAmount += Time.deltaTime;
            }
        }
        else
        {
            //Debug.Log(leftGaze.isValid + ", " + rightGaze.isValid);
        }
    }
}
