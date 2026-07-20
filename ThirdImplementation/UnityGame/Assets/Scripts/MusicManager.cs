using UnityEngine;

public class MusicManager : MonoBehaviour
{
    public AudioClip lockdownMusic;
    AudioSource audiosource;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        audiosource = GetComponent<AudioSource>();
        GameManagerScript.alarmInitiated += OnAlarm;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnAlarm()
    {
        audiosource.Stop();
        audiosource.clip = lockdownMusic;
        audiosource.volume = 0.25f;
        audiosource.Play();
    }

}
