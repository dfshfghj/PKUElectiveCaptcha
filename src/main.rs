use anyhow::{anyhow, bail, Context, Result};
use chrono::Utc;
use clap::{Parser, Subcommand};
use image::{imageops::FilterType, DynamicImage};
use ndarray::Array4;
use ort::{session::Session, value::TensorRef};
use reqwest::blocking::{Client, Response};
use reqwest::header::{CONTENT_TYPE, COOKIE, REFERER, SET_COOKIE, USER_AGENT};
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
const USER_AGENT_VALUE: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/98 Safari/537.36";
const CHARSET: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
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
        #[arg(long)] username: Option<String>,
        #[arg(long)] password: Option<String>,
        #[arg(long)] channel: Option<String>,
        #[arg(long, default_value_t = 20)] count: usize,
        #[arg(long, default_value_t = 0.8)] delay: f64,
        #[arg(long, default_value = "data/collected")] output_dir: PathBuf,
        #[arg(long, default_value = ".session_cookies.json")] cookies_file: PathBuf,
        #[arg(long, default_value = "data/captcha_mobilenet_ctc.onnx")] model: PathBuf,
        #[arg(long)] reuse_cookies: bool,
        #[arg(long)] images_only: bool,
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
}

impl HttpSession {
    fn new(cookies: HashMap<String, String>) -> Result<Self> {
        let client = Client::builder()
            .danger_accept_invalid_certs(true)
            .cookie_store(true)
            .timeout(Duration::from_secs(20))
            .build()?;
        Ok(Self { client, cookies })
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
            let Some(pair) = value.split(';').next() else { continue };
            let Some((name, value)) = pair.split_once('=') else { continue };
            self.cookies.insert(name.trim().to_string(), value.trim().to_string());
        }
        for cookie in response.cookies() {
            self.cookies.insert(cookie.name().to_string(), cookie.value().to_string());
        }
    }

    fn get(&mut self, url: &str, query: &[(&str, &str)]) -> Result<Response> {
        let cookie = self.cookie_header();
        let mut last_error = None;
        for attempt in 0..3 {
            let result = self.client.get(url)
                .query(query)
                .header(USER_AGENT, USER_AGENT_VALUE)
                .header(REFERER, HOME_URL)
                .header(COOKIE, cookie.clone())
                .send();
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
        let response = self.client.post(url)
            .form(form)
            .header(USER_AGENT, USER_AGENT_VALUE)
            .header(REFERER, HOME_URL)
            .header(COOKIE, cookie)
            .send()?;
        self.record_cookies(&response);
        Ok(response)
    }
}

fn authenticate(session: &mut HttpSession, username: &str, password: &str, channel: Option<&str>) -> Result<()> {
    let response = session.post_form(LOGIN_URL, &[
        ("appid", "syllabus"), ("userName", username), ("password", password),
        ("randCode", ""), ("smsCode", ""), ("otpCode", ""),
        ("redirUrl", "http://elective.pku.edu.cn:80/elective2008/agent4Iaaa.jsp/../ssoLogin.do"),
    ])?;
    let payload: serde_json::Value = response.json()?;
    if payload.get("success").and_then(|value| value.as_bool()) != Some(true) {
        bail!("login failed: {payload}");
    }
    let token = payload.get("token").and_then(|value| value.as_str()).context("login token missing")?;
    let response = session.get(SSO_URL, &[("rand", "0.1"), ("token", token)])?;
    let body = response.text()?;
    if body.contains(HELP_TITLE) { return Ok(()); }
    if body.contains("/scnStAthVef.jsp/") {
        let channel = channel.context("identity selection required; pass --channel bzx or --channel bfx")?;
        let sida = body.split("/ssoLogin.do?sida=").nth(1)
            .and_then(|value| value.split('&').next())
            .filter(|value| !value.is_empty())
            .context("unable to extract sida from SSO response")?;
        let response = session.get(SSO_URL, &[("sida", sida), ("sttp", channel)])?;
        if response.text()?.contains(HELP_TITLE) { return Ok(()); }
    }
    bail!("after login check did not reach elective home")
}

fn save_cookies(path: &Path, cookies: &HashMap<String, String>) -> Result<()> {
    let file = CookieFile { cookies: cookies.clone() };
    fs::write(path, serde_json::to_vec_pretty(&file)?)?;
    Ok(())
}

fn load_cookies(path: &Path) -> Result<HashMap<String, String>> {
    if !path.exists() { return Ok(HashMap::new()); }
    Ok(serde_json::from_slice::<CookieFile>(&fs::read(path)?)?.cookies)
}

fn fetch_captcha(session: &mut HttpSession) -> Result<Vec<u8>> {
    let response = session.get(CAPTCHA_URL, &[("Rand", "0.1")])?;
    if !response.status().is_success() { bail!("captcha request failed: {}", response.status()); }
    if response.headers().get(CONTENT_TYPE).and_then(|value| value.to_str().ok())
        .map(|value| value.to_ascii_lowercase().starts_with("text/html")) == Some(true) {
        bail!("captcha response is HTML; login session has expired");
    }
    let body = response.bytes()?.to_vec();
    image::load_from_memory(&body).context("captcha response is not a valid image; login session has expired")?;
    Ok(body)
}

struct CaptchaModel {
    session: Session,
}

impl CaptchaModel {
    fn load(path: &Path) -> Result<Self> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(Self { session })
    }

    fn input(image_bytes: &[u8]) -> Result<Array4<f32>> {
        let image = image::load_from_memory(image_bytes)?.to_luma8();
        let image = DynamicImage::ImageLuma8(image).resize_exact(130, 52, FilterType::Triangle).to_luma8();
        let values = image.pixels().map(|pixel| (f32::from(pixel[0]) / 255.0 - 0.5) / 0.5).collect::<Vec<_>>();
        Ok(Array4::from_shape_vec((1, 1, 52, 130), values)?)
    }

    fn predict(&mut self, image_bytes: &[u8]) -> Result<String> {
        let input = Self::input(image_bytes)?;
        let outputs = self.session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
        let (_, logits) = outputs[0].try_extract_tensor::<f32>()?;
        let mut beam: HashMap<String, (f64, f64)> = HashMap::from([(String::new(), (1.0, 0.0))]);
        for timestep in 0..10 {
            let mut distribution = Vec::with_capacity(37);
            for index in 0..37 { distribution.push(logits[timestep * 37 + index]); }
            let max = distribution.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp = distribution.iter().map(|value| f64::from((*value - max).exp())).collect::<Vec<_>>();
            let total = exp.iter().sum::<f64>();
            let probabilities = exp.into_iter().map(|value| value / total).collect::<Vec<_>>();
            let mut indices = (0..37).collect::<Vec<_>>();
            indices.sort_by(|a, b| probabilities[*b].partial_cmp(&probabilities[*a]).unwrap());
            let mut next: HashMap<String, (f64, f64)> = HashMap::new();
            for (prefix, (blank, text)) in &beam {
                for index in indices.iter().take(20) {
                    let probability = probabilities[*index];
                    if *index == 0 {
                        let scores = next.entry(prefix.clone()).or_default();
                        scores.0 += (blank + text) * probability;
                    } else {
                        let character = CHARSET[*index - 1] as char;
                        if prefix.ends_with(character) {
                            let same = next.entry(prefix.clone()).or_default();
                            same.1 += text * probability;
                            let extended = next.entry(format!("{prefix}{character}")).or_default();
                            extended.1 += blank * probability;
                        } else {
                            let extended = next.entry(format!("{prefix}{character}")).or_default();
                            extended.1 += (blank + text) * probability;
                        }
                    }
                }
            }
            let mut ranked = next.into_iter().collect::<Vec<_>>();
            ranked.sort_by(|a, b| (b.1.0 + b.1.1).partial_cmp(&(a.1.0 + a.1.1)).unwrap());
            beam = ranked.into_iter().take(5).collect();
        }
        beam.into_iter().max_by(|a, b| (a.1.0 + a.1.1).partial_cmp(&(b.1.0 + b.1.1)).unwrap())
            .map(|(label, _)| label)
            .filter(|label| !label.is_empty())
            .ok_or_else(|| anyhow!("OCR beam search produced no candidate"))
    }
}

fn sample_name(index: usize, bytes: &[u8]) -> String {
    let mut digest = Sha256::new(); digest.update(bytes);
    let hash = format!("{:x}", digest.finalize());
    format!("{}_{index:05}_{}.png", Utc::now().format("%Y%m%dT%H%M%S%3fZ"), &hash[..12])
}

fn append_manifest(path: &Path, filename: &str, status: &str, prediction: &str, verified: bool, bytes: &[u8], response: &str) -> Result<()> {
    let new_file = !path.exists();
    let mut writer = csv::WriterBuilder::new().has_headers(false).from_writer(fs::OpenOptions::new().create(true).append(true).open(path)?);
    if new_file { writer.write_record(["timestamp", "filename", "status", "predicted_label", "ocr_source", "verified", "sha256", "bytes", "verification_response"])?; }
    let mut digest = Sha256::new(); digest.update(bytes);
    writer.write_record([
        Utc::now().to_rfc3339(), filename.to_string(), status.to_string(), prediction.to_string(),
        "trained_model".to_string(), verified.to_string(), format!("{:x}", digest.finalize()),
        bytes.len().to_string(), response.to_string(),
    ])?;
    writer.flush()?;
    Ok(())
}

fn collect(args: &Command) -> Result<()> {
    let Command::Collect { username, password, channel, count, delay, output_dir, cookies_file, model, reuse_cookies, images_only } = args;
    let username = username.clone().or_else(|| std::env::var("HEED_USERNAME").ok()).context("missing username")?;
    let password = password.clone().or_else(|| std::env::var("HEED_PASSWORD").ok()).or_else(|| { print!("password: "); io::stdout().flush().ok(); let mut value=String::new(); io::stdin().read_line(&mut value).ok()?; Some(value.trim().to_string()) }).filter(|value| !value.is_empty()).context("missing password")?;
    let cookies = if *reuse_cookies { load_cookies(cookies_file)? } else { HashMap::new() };
    let mut session = HttpSession::new(cookies)?;
    fs::create_dir_all(output_dir.join("raw"))?;
    fs::create_dir_all(output_dir.join("auto_labeled"))?;
    fs::create_dir_all(output_dir.join("review"))?;
    let mut model_instance = if *images_only { None } else { Some(CaptchaModel::load(model)?) };
    authenticate(&mut session, &username, &password, channel.as_deref())?;
    save_cookies(cookies_file, &session.cookies)?;
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
            let response = session.post_form(VERIFY_URL, &[("validCode", &prediction), ("xh", &username)])?;
            let payload = response.text()?;
            let verified = payload.contains("\"valid\":\"2\"") || payload.contains("\"valid\":  \"2\"");
            if verified { correct += 1; }
            let status = if verified { "auto_verified" } else { "needs_manual_label" };
            let relative = if verified { format!("auto_labeled/{}_{}.png", name.trim_end_matches(".png"), prediction) } else { format!("review/{name}") };
            fs::rename(&raw_path, output_dir.join(&relative))?;
            append_manifest(&output_dir.join("manifest.csv"), &relative, status, &prediction, verified, &image, &payload)?;
            println!("[{index}/{count}] {status} label={prediction} correct={correct}/{index} ({:.2}%)", correct as f64 / index as f64 * 100.0);
        }
        if index < *count {
            if !*images_only && index % BATCH_SIZE == 0 {
                thread::sleep(Duration::from_secs(5));
                session = HttpSession::new(HashMap::new())?;
                authenticate(&mut session, &username, &password, channel.as_deref())?;
                save_cookies(cookies_file, &session.cookies)?;
            } else if *delay > 0.0 { thread::sleep(Duration::from_secs_f64(*delay)); }
        }
    }
    if !*images_only { println!("summary: correct={correct}/{count} ({:.2}%)", correct as f64 / *count as f64 * 100.0); }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    collect(&cli.command)
}
