use anyhow::{Context, Result, bail};
use chrono::Utc;
use clap::{Parser, Subcommand};
use image::{DynamicImage, imageops::FilterType};
use reqwest::blocking::{Client, Response};
use reqwest::header::{
    ACCEPT, CACHE_CONTROL, CONTENT_TYPE, COOKIE, REFERER, SET_COOKIE, USER_AGENT,
};
use rten::Model;
use rten_tensor::{AsView, NdTensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

const LOGIN_URL: &str = "https://iaaa.pku.edu.cn/iaaa/oauthlogin.do";
const SSO_URL: &str = "https://elective.pku.edu.cn/elective2008/ssoLogin.do";
const HOME_URL: &str = "https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/help/HelpController.jpf";
const CAPTCHA_URL: &str = "https://elective.pku.edu.cn/elective2008/DrawServlet";
const VERIFY_URL: &str = "https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/supplement/validate.do";
const HELP_TITLE: &str = "<title>帮助-总体流程</title>";
const USER_AGENT_VALUE: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/98 Safari/537.36";
const CACHE_CONTROL_VALUE: &str = "max-age=0";
const BATCH_SIZE: usize = 15;

#[derive(Parser)]
#[command(name = "pku-elective-captcha-collector")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Collect {
        #[arg(long)]
        username: Option<String>,
        #[arg(long)]
        password: Option<String>,
        #[arg(long)]
        channel: Option<String>,
        #[arg(long, default_value_t = 20)]
        count: usize,
        #[arg(long, default_value_t = 0.8)]
        delay: f64,
        #[arg(long, default_value = "data/collected")]
        output_dir: PathBuf,
        #[arg(long, default_value = ".session_cookies.json")]
        cookies_file: PathBuf,
        #[arg(long, default_value = "data/recognizer.rten")]
        model: PathBuf,
        #[arg(long)]
        reuse_cookies: bool,
        #[arg(long)]
        images_only: bool,
    },
}

#[derive(Default, Serialize, Deserialize)]
struct CookieFile {
    #[serde(default)]
    cookies: HashMap<String, String>,
}

struct HttpSession {
    client: Client,
    cookies: HashMap<String, String>,
    manual_cookie_header: bool,
}

impl HttpSession {
    fn new(cookies: HashMap<String, String>) -> Result<Self> {
        let client = Client::builder()
            .danger_accept_invalid_certs(true)
            // The elective server is an old HTTP/1.1 Java application. Keep
            // the transport identical to Python requests; HTTP/2 can make
            // its session affinity fail even when JSESSIONID is present.
            .http1_only()
            .cookie_store(true)
            .timeout(Duration::from_secs(20))
            .build()?;
        let manual_cookie_header = !cookies.is_empty();
        Ok(Self {
            client,
            cookies,
            manual_cookie_header,
        })
    }

    fn cookie_header(&self) -> String {
        self.cookies
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join("; ")
    }

    fn record_cookies(&mut self, response: &Response) {
        for value in response.headers().get_all(SET_COOKIE).iter() {
            let Ok(value) = value.to_str() else { continue };
            let Some(pair) = value.split(';').next() else {
                continue;
            };
            let Some((name, value)) = pair.split_once('=') else {
                continue;
            };
            self.cookies
                .insert(name.trim().to_string(), value.trim().to_string());
        }
        for cookie in response.cookies() {
            self.cookies
                .insert(cookie.name().to_string(), cookie.value().to_string());
        }
    }

    fn get(&mut self, url: &str, query: &[(&str, &str)]) -> Result<Response> {
        let cookie = self.cookie_header();
        let mut last_error = None;
        for attempt in 0..3 {
            let mut request = self
                .client
                .get(url)
                .query(query)
                .header(USER_AGENT, USER_AGENT_VALUE)
                .header(REFERER, HOME_URL)
                .header(CACHE_CONTROL, CACHE_CONTROL_VALUE)
                .header(ACCEPT, "*/*");
            // Do not send an empty Cookie header. Python requests omits it
            // when the session has no cookies, which matters to the SSO flow.
            if self.manual_cookie_header && !cookie.is_empty() {
                request = request.header(COOKIE, cookie.clone());
            }
            let result = request.send();
            match result {
                Ok(response) => {
                    self.record_cookies(&response);
                    return Ok(response);
                }
                Err(error) if attempt < 2 => {
                    last_error = Some(error);
                    thread::sleep(Duration::from_millis(500 * (attempt + 1)));
                }
                Err(error) => return Err(error.into()),
            }
        }
        Err(last_error.expect("retry loop always has an error").into())
    }

    fn post_form(&mut self, url: &str, form: &[(&str, &str)]) -> Result<Response> {
        let cookie = self.cookie_header();
        let mut request = self
            .client
            .post(url)
            .form(form)
            .header(USER_AGENT, USER_AGENT_VALUE)
            .header(REFERER, HOME_URL)
            .header(CACHE_CONTROL, CACHE_CONTROL_VALUE)
            .header(ACCEPT, "*/*");
        if self.manual_cookie_header && !cookie.is_empty() {
            request = request.header(COOKIE, cookie.clone());
        }
        let response = request.send()?;
        self.record_cookies(&response);
        Ok(response)
    }
}

fn verify_alive(session: &mut HttpSession) -> Result<bool> {
    let response = session.get(HOME_URL, &[])?;
    Ok(response.status().is_success() && response.text()?.contains(HELP_TITLE))
}

fn authenticate(
    session: &mut HttpSession,
    username: &str,
    password: &str,
    channel: Option<&str>,
) -> Result<()> {
    let login_cookie = format!("userName={username}");
    let response = session
        .client
        .post(LOGIN_URL)
        .form(&[
            ("appid", "syllabus"),
            ("userName", username),
            ("password", password),
            ("randCode", ""),
            ("smsCode", ""),
            ("otpCode", ""),
            (
                "redirUrl",
                "http://elective.pku.edu.cn:80/elective2008/agent4Iaaa.jsp/../ssoLogin.do",
            ),
        ])
        .header(USER_AGENT, USER_AGENT_VALUE)
        .header(REFERER, HOME_URL)
        .header(CACHE_CONTROL, CACHE_CONTROL_VALUE)
        .header(ACCEPT, "*/*")
        .header(COOKIE, login_cookie)
        .send()?;
    session.record_cookies(&response);
    let payload: serde_json::Value = response.json()?;
    if payload.get("success").and_then(|value| value.as_bool()) != Some(true) {
        bail!("login failed: {payload}");
    }
    let token = payload
        .get("token")
        .and_then(|value| value.as_str())
        .context("login token missing")?;
    let response = session.get(SSO_URL, &[("rand", "0.1"), ("token", token)])?;
    let body = response.text()?;
    if body.contains(HELP_TITLE) {
        return Ok(());
    }
    if body.contains("/scnStAthVef.jsp/") {
        let channel =
            channel.context("identity selection required; pass --channel bzx or --channel bfx")?;
        let sida = body
            .split("/ssoLogin.do?sida=")
            .nth(1)
            .and_then(|value| value.split('&').next())
            .filter(|value| !value.is_empty())
            .context("unable to extract sida from SSO response")?;
        let response = session.get(SSO_URL, &[("sida", sida), ("sttp", channel)])?;
        if response.text()?.contains(HELP_TITLE) {
            return Ok(());
        }
    }
    bail!("after login check did not reach elective home")
}

fn save_cookies(path: &Path, cookies: &HashMap<String, String>) -> Result<()> {
    let file = CookieFile {
        cookies: cookies.clone(),
    };
    fs::write(path, serde_json::to_vec_pretty(&file)?)?;
    Ok(())
}

fn load_cookies(path: &Path) -> Result<HashMap<String, String>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    Ok(serde_json::from_slice::<CookieFile>(&fs::read(path)?)?.cookies)
}

fn fetch_captcha(session: &mut HttpSession) -> Result<Vec<u8>> {
    let response = session.get(CAPTCHA_URL, &[("Rand", "0.1")])?;
    if !response.status().is_success() {
        bail!("captcha request failed: {}", response.status());
    }
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("<missing>")
        .to_string();
    let body = response.bytes()?.to_vec();
    if content_type.to_ascii_lowercase().starts_with("text/html") {
        bail!("captcha response is HTML; login session has expired");
    }
    image::load_from_memory(&body)
        .context("captcha response is not a valid image; login session has expired")?;
    Ok(body)
}

#[derive(Deserialize)]
struct CharsetConfig {
    charset: Vec<String>,
    image: [i32; 2],
    word: bool,
    channel: u8,
}

struct CaptchaModel {
    model: Model,
    input_id: rten::NodeId,
    output_id: rten::NodeId,
    charset: Vec<String>,
}

impl CaptchaModel {
    fn load(path: &Path) -> Result<Self> {
        let model = Model::load_file(path)
            .with_context(|| format!("failed to load RTen model {}", path.display()))?;
        let charset_path = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("charsets.json");
        let config: CharsetConfig = serde_json::from_slice(&fs::read(&charset_path)?)?;
        if config.image != [-1, 64] || config.word || config.channel != 3 {
            bail!(
                "unsupported ddddocr charset config in {}: image={:?}, word={}, channel={}",
                charset_path.display(),
                config.image,
                config.word,
                config.channel
            );
        }
        if config.charset.is_empty() || config.charset[0] != " " {
            bail!("ddddocr charset must reserve charset[0] as the CTC blank token");
        }
        let input_id = model.node_id("input1")?;
        let output_id = model.node_id("output")?;
        Ok(Self {
            model,
            input_id,
            output_id,
            charset: config.charset,
        })
    }

    fn input(image_bytes: &[u8]) -> Result<NdTensor<f32, 4>> {
        let image = image::load_from_memory(image_bytes)?.to_rgb8();
        let width = ((image.width() as f32 * 64.0) / image.height() as f32) as u32;
        let image = DynamicImage::ImageRgb8(image)
            .resize_exact(width.max(1), 64, FilterType::Lanczos3)
            .to_rgb8();
        let width = image.width() as usize;
        let height = image.height() as usize;
        let plane_size = width * height;
        let mean = [0.485_f32, 0.456, 0.406];
        let std = [0.229_f32, 0.224, 0.225];
        let mut values = vec![0.0_f32; 3 * plane_size];
        for (pixel_index, pixel) in image.pixels().enumerate() {
            for channel in 0..3 {
                values[channel * plane_size + pixel_index] =
                    (f32::from(pixel[channel]) / 255.0 - mean[channel]) / std[channel];
            }
        }
        Ok(NdTensor::from_data([1, 3, height, width], values))
    }

    fn predict(&mut self, image_bytes: &[u8]) -> Result<String> {
        let input = Self::input(image_bytes)?;
        let outputs =
            self.model
                .run_n(vec![(self.input_id, input.into())], [self.output_id], None)?;
        let tokens: NdTensor<i32, 2> = outputs[0].clone().try_into()?;
        let mut result = String::new();
        let mut last_token = 0_i32;
        for &token in tokens.iter() {
            if token == last_token {
                continue;
            }
            last_token = token;
            if token == 0 {
                continue;
            }
            let character = self
                .charset
                .get(token as usize)
                .with_context(|| format!("OCR token {token} is outside the charset"))?;
            result.push_str(character);
        }
        if result.is_empty() {
            bail!("OCR produced an empty CTC token sequence");
        }
        Ok(result)
    }
}

fn sample_name(index: usize, bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    let hash = format!("{:x}", digest.finalize());
    format!(
        "{}_{index:05}_{}.png",
        Utc::now().format("%Y%m%dT%H%M%S%3fZ"),
        &hash[..12]
    )
}

fn append_manifest(
    path: &Path,
    filename: &str,
    status: &str,
    prediction: &str,
    verified: bool,
    bytes: &[u8],
    response: &str,
) -> Result<()> {
    let new_file = !path.exists();
    let mut writer = csv::WriterBuilder::new().has_headers(false).from_writer(
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?,
    );
    if new_file {
        writer.write_record([
            "timestamp",
            "filename",
            "status",
            "predicted_label",
            "ocr_source",
            "verified",
            "sha256",
            "bytes",
            "verification_response",
        ])?;
    }
    let mut digest = Sha256::new();
    digest.update(bytes);
    writer.write_record([
        Utc::now().to_rfc3339(),
        filename.to_string(),
        status.to_string(),
        prediction.to_string(),
        "trained_model".to_string(),
        verified.to_string(),
        format!("{:x}", digest.finalize()),
        bytes.len().to_string(),
        response.to_string(),
    ])?;
    writer.flush()?;
    Ok(())
}

fn collect(args: &Command) -> Result<()> {
    let Command::Collect {
        username,
        password,
        channel,
        count,
        delay,
        output_dir,
        cookies_file,
        model,
        reuse_cookies,
        images_only,
    } = args;
    let username = username
        .clone()
        .or_else(|| std::env::var("HEED_USERNAME").ok())
        .context("missing username")?;
    let password = password
        .clone()
        .or_else(|| std::env::var("HEED_PASSWORD").ok())
        .or_else(|| {
            print!("password: ");
            io::stdout().flush().ok();
            let mut value = String::new();
            io::stdin().read_line(&mut value).ok()?;
            Some(value.trim().to_string())
        })
        .filter(|value| !value.is_empty())
        .context("missing password")?;
    let loaded_cookies = if *reuse_cookies {
        load_cookies(cookies_file)?
    } else {
        HashMap::new()
    };
    let reuse_loaded = *reuse_cookies && !loaded_cookies.is_empty();
    let mut session = HttpSession::new(loaded_cookies)?;
    fs::create_dir_all(output_dir.join("raw"))?;
    fs::create_dir_all(output_dir.join("auto_labeled"))?;
    fs::create_dir_all(output_dir.join("review"))?;
    let mut model_instance = if *images_only {
        None
    } else {
        Some(CaptchaModel::load(model)?)
    };
    if !reuse_loaded || !verify_alive(&mut session)? {
        authenticate(&mut session, &username, &password, channel.as_deref())?;
        save_cookies(cookies_file, &session.cookies)?;
    }
    let mut correct = 0usize;
    for index in 1..=*count {
        let image = fetch_captcha(&mut session)?;
        let name = sample_name(index, &image);
        let raw_path = output_dir.join("raw").join(&name);
        fs::write(&raw_path, &image)?;
        if *images_only {
            println!("[{index}/{count}] collected: raw/{name}");
        } else {
            let prediction = model_instance.as_mut().unwrap().predict(&image)?;
            let response =
                session.post_form(VERIFY_URL, &[("validCode", &prediction), ("xh", &username)])?;
            let payload = response.text()?;
            let verified =
                payload.contains("\"valid\":\"2\"") || payload.contains("\"valid\":  \"2\"");
            if verified {
                correct += 1;
            }
            let status = if verified {
                "auto_verified"
            } else {
                "needs_manual_label"
            };
            let relative = if verified {
                format!(
                    "auto_labeled/{}_{}.png",
                    name.trim_end_matches(".png"),
                    prediction
                )
            } else {
                format!("review/{name}")
            };
            fs::rename(&raw_path, output_dir.join(&relative))?;
            append_manifest(
                &output_dir.join("manifest.csv"),
                &relative,
                status,
                &prediction,
                verified,
                &image,
                &payload,
            )?;
            println!(
                "[{index}/{count}] {status} label={prediction} correct={correct}/{index} ({:.2}%)",
                correct as f64 / index as f64 * 100.0
            );
        }
        if index < *count {
            if !*images_only && index % BATCH_SIZE == 0 {
                thread::sleep(Duration::from_secs(5));
                session = HttpSession::new(HashMap::new())?;
                authenticate(&mut session, &username, &password, channel.as_deref())?;
                save_cookies(cookies_file, &session.cookies)?;
            } else if *delay > 0.0 {
                thread::sleep(Duration::from_secs_f64(*delay));
            }
        }
    }
    if !*images_only {
        println!(
            "summary: correct={correct}/{count} ({:.2}%)",
            correct as f64 / *count as f64 * 100.0
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    collect(&cli.command)
}
